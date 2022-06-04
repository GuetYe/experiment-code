# _*_ coding:utf-8_*

from ryu import cfg
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.base.app_manager import lookup_service_brick
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ipv4
from ryu.lib.packet import arp
from collections import defaultdict
from utils_file import *
from thread import start_new_thread, get_ident

CONF = cfg.CONF


class OSPFForwarding(app_manager.RyuApp):
    """
        ShortestForwarding is a Ryu app for forwarding packets in shortest
        path.
        The shortest path computation is done by module network awareness,
        network monitor and network delay detector.
    """

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(OSPFForwarding, self).__init__(*args, **kwargs)
        self.name = "ospf_algorithm"
        self.topology_module = lookup_service_brick("topology_module")
        self.monitor_module_module = lookup_service_brick("status_module")
        self.delay_module = lookup_service_brick("delay_module")
        self.management_module = lookup_service_brick("management_module")
        self.send_flow_flag = False  # the flag of send flow      .
        self.tos_list_count = []
        self.tos_list_flag = [1]
        self.clear_tos_flag = False
        self.count = 0
        self.pre_tos = 0
        self.per_tos_end = None

        # --- evaluating indicator --- #
        self.ospf_link_path = defaultdict(dict)
        self.ospf_link_delay = defaultdict(dict)
        self.ospf_link_loss = defaultdict(dict)
        self.ospf_link_throughput = defaultdict(dict)
        self.json_obj = SaveInfosToJson()
        self.path_obj = self.json_obj.make_path

    def check_metric_is_format(self, metric_dict):
        """

        :param metric_dict:
        :return:
        """

        # --- check metric infos --- #
        inner_metric_len = 0
        for k1 in metric_dict.keys():
            inner_metric_len += len(metric_dict[k1].keys())

        return inner_metric_len

    def save_all_link_indicators(self):
        """
            Save all indicators to json file.
        :return:
        """
        if self.check_metric_is_format(self.ospf_link_path) == CF.NODES * (CF.NODES - 1):
            self.json_obj.save_all_path_infos(self.ospf_link_path, type="OSPF_PATH")
            self.json_obj.save_all_metrics_infos(self.ospf_link_throughput, type="OSPF_THROUGHTPUT_PATH")
            self.json_obj.save_all_metrics_infos(self.ospf_link_delay, type="OSPF_DELAY_PATH")
            self.json_obj.save_all_metrics_infos(self.ospf_link_loss, type="OSPF_LOSS_PATH")
            self.ospf_link_path.clear()
            self.ospf_link_throughput.clear()
            self.ospf_link_delay.clear()
            self.ospf_link_loss.clear()
            print("\033[35;1m ospf [%s] done !!!\033[0m" % get_ident())

    def add_flow(self, dp, p, match, actions, idle_timeout=0, hard_timeout=0):
        """
            Add flow.
        :param dp:
        :param p:
        :param match:
        :param actions:
        :param idle_timeout:
        :param hard_timeout:
        :return:
        """
        ofproto = dp.ofproto
        parser = dp.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]

        mod = parser.OFPFlowMod(datapath=dp, priority=p,
                                idle_timeout=idle_timeout,
                                hard_timeout=hard_timeout,
                                match=match, instructions=inst)
        dp.send_msg(mod)

    def send_flow_mod(self, tos, datapath, flow_info, src_port, dst_port):
        """

        :param tos:
        :param datapath:
        :param flow_info:
        :param src_port:
        :param dst_port:
        :return:
        """
        parser = datapath.ofproto_parser
        actions = []
        actions.append(parser.OFPActionOutput(dst_port))
        tos_dscp = (tos & 0b11111100) / 4
        tos_ecn = tos & 0b00000011

        match = parser.OFPMatch(
            in_port=src_port, eth_type=flow_info[0],
            ipv4_src=flow_info[1], ipv4_dst=flow_info[2], ip_dscp=tos_dscp, ip_ecn=tos_ecn)

        self.add_flow(datapath, 1, match, actions,
                      idle_timeout=15, hard_timeout=60)

    def _build_packet_out(self, datapath, buffer_id, src_port, dst_port, data):
        """

        :param datapath:
        :param buffer_id:
        :param src_port:
        :param dst_port:
        :param data:
        :return:
        """
        actions = []
        if dst_port:
            actions.append(datapath.ofproto_parser.OFPActionOutput(dst_port))
        msg_data = None
        if buffer_id == datapath.ofproto.OFP_NO_BUFFER:
            if data is None:
                return None
            msg_data = data

        out = datapath.ofproto_parser.OFPPacketOut(
            datapath=datapath, buffer_id=buffer_id,
            data=msg_data, in_port=src_port, actions=actions)
        return out

    def send_packet_out(self, datapath, buffer_id, src_port, dst_port, data):
        """

        :param datapath:
        :param buffer_id:
        :param src_port:
        :param dst_port:
        :param data:
        :return:
        """
        out = self._build_packet_out(datapath, buffer_id,
                                     src_port, dst_port, data)
        if out:
            datapath.send_msg(out)

    def flood(self, msg):
        """
            flood ARP packet to the access port which has no record of host.
        :param msg:
        :return:
        """
        datapath = msg.datapath
        ofproto = datapath.ofproto
        for dpid in self.topology_module.all_host_port:
            for port in self.topology_module.all_host_port[dpid]:
                if (dpid, port) not in self.topology_module.mapping_ip_mac.keys():
                    datapath = self.topology_module.datapaths[dpid]
                    out = self._build_packet_out(
                        datapath, ofproto.OFP_NO_BUFFER,
                        ofproto.OFPP_CONTROLLER, port, msg.data)
                    datapath.send_msg(out)
        self.logger.debug("Flooding msg")

    def install_flow(self, tos, datapaths, path, flow_info, buffer_id, data=None):
        """
            Install flow entires for roundtrip: go and back.
        :param datapaths:
        :param path:
        :param flow_info:
        :param buffer_id:
        :param data:
        :return:
        """
        if path is None or len(path) == 0:
            self.logger.info("Path error!")
            return
        in_port = flow_info[3]
        first_dp = datapaths[path[0]]
        out_port = first_dp.ofproto.OFPP_LOCAL
        back_info = (flow_info[0], flow_info[2], flow_info[1])

        # inter_link
        if len(path) > 2:
            for i in xrange(1, len(path) - 1):
                port = self.topology_module.get_port_pair_from_link(path[i - 1], path[i])
                port_next = self.topology_module.get_port_pair_from_link(path[i], path[i + 1])
                if port and port_next:
                    src_port, dst_port = port[1], port_next[0]
                    datapath = datapaths[path[i]]
                    self.send_flow_mod(tos, datapath, flow_info, src_port, dst_port)
                    self.send_flow_mod(tos, datapath, back_info, dst_port, src_port)
                    self.send_packet_out(datapath, buffer_id, src_port, dst_port, data)
                    self.logger.debug("inter_link flow install")
        if len(path) > 1:
            # the last flow entry: tor -> host
            port_pair = self.topology_module.get_port_pair_from_link(path[-2], path[-1])
            if port_pair is None:
                self.logger.info("Port is not found")
                return
            src_port = port_pair[1]

            dst_port = self.topology_module.get_dst_port_from_mapping(flow_info[2])
            if dst_port is None:
                self.logger.info("Last port is not found.")
                return

            last_dp = datapaths[path[-1]]
            self.send_flow_mod(tos, last_dp, flow_info, src_port, dst_port)
            self.send_flow_mod(tos, last_dp, back_info, dst_port, src_port)
            self.send_packet_out(last_dp, buffer_id, src_port, dst_port, data)

            # the first flow entry
            port_pair = self.topology_module.get_port_pair_from_link(path[0], path[1])
            if port_pair is None:
                self.logger.info("Port not found in first hop.")
                return
            out_port = port_pair[0]
            self.send_flow_mod(tos, first_dp, flow_info, in_port, out_port)
            self.send_flow_mod(tos, first_dp, back_info, out_port, in_port)
            self.send_packet_out(first_dp, buffer_id, in_port, out_port, data)

        # src and dst on the same datapath
        else:
            out_port = self.topology_module.get_dst_port_from_mapping(flow_info[2])
            if out_port is None:
                self.logger.info("Out_port is None in same dp")
                return
            self.send_flow_mod(tos, first_dp, flow_info, in_port, out_port)
            self.send_flow_mod(tos, first_dp, back_info, out_port, in_port)
            self.send_packet_out(first_dp, buffer_id, in_port, out_port, data)

    def optimal_routing_forwarding(self, tos, msg, eth_type, ip_src, ip_dst):
        """
                Get optimal routing forwarding path.
        :param msg:
        :param eth_type:
        :param ip_src:
        :param ip_dst:
        :return:
        """
        datapath = msg.datapath
        in_port = msg.match["in_port"]
        result = self.topology_module.get_sw_from_mapping(datapath.id, in_port, ip_src, ip_dst)
        if result:
            src_sw, dst_sw = result[0], result[1]
            if dst_sw:
                # same #
                if src_sw == dst_sw:
                    path = [src_sw]
                # dijkstra algo #
                else:
                    path = self.topology_module.get_ospf_path(src_sw, dst_sw)
                flow_info = (eth_type, ip_src, ip_dst, in_port)
                self.install_flow(tos, self.topology_module.datapaths, path,
                                  flow_info, msg.buffer_id, msg.data)
                if tos != 0:
                    self.count += 1
                    print("<tos> :", tos, " path : ", path)
                    delay, loss, throughput = self.management_module.get_network_performance([path])
                    self.ospf_link_path[str(src_sw)][str(dst_sw)] = path
                    self.ospf_link_delay[str(src_sw)][str(dst_sw)] = delay
                    self.ospf_link_loss[str(src_sw)][str(dst_sw)] = loss
                    self.ospf_link_throughput[str(src_sw)][str(dst_sw)] = throughput
                    if self.count == self.per_tos_end:
                        start_new_thread(self.save_all_link_indicators, )
        return

    def arp_forwarding(self, msg, src_ip, dst_ip):
        """
            Send ARP packet to the destination host, if the dst host record is existed,
            else, flow it to the unknow access port(flood).
        :param msg:
        :param src_ip:
        :param dst_ip:
        :return:
        """
        datapath = msg.datapath
        ofproto = datapath.ofproto
        if self.topology_module is None:
            self.topology_module = lookup_service_brick("topology_module")
        result = self.topology_module.get_dpid_port_from_mapping(dst_ip)
        if result:  # host record in access table.
            datapath_dst, out_port = result[0], result[1]
            datapath = self.topology_module.datapaths[datapath_dst]
            # send packet out message to controller
            out = self._build_packet_out(datapath, ofproto.OFP_NO_BUFFER,
                                         ofproto.OFPP_CONTROLLER,
                                         out_port, msg.data)
            datapath.send_msg(out)
            self.logger.debug("Reply ARP to knew host")
        else:
            self.flood(msg)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """
            handler various message, arp, ipv4
        :param ev:
        :return:
        """
        msg = ev.msg
        pkt = packet.Packet(msg.data)
        arp_pkt = pkt.get_protocol(arp.arp)
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if self.topology_module is None:
            self.topology_module = lookup_service_brick("topology_module")

        # handler arp pkt #
        if isinstance(arp_pkt, arp.arp):
            self.logger.debug("ARP processing")
            self.arp_forwarding(msg, arp_pkt.src_ip, arp_pkt.dst_ip)

        # handler ip pkt #
        if isinstance(ip_pkt, ipv4.ipv4):
            self.logger.debug("IPV4 processing")
            if len(pkt.get_protocols(ethernet.ethernet)):
                eth_type = pkt.get_protocols(ethernet.ethernet)[0].ethertype
                tos = ip_pkt.tos
                if tos == CF.DROP_TOS: return
                # arp
                if tos == 0 and self.topology_module.arp_pkt_detect:
                    self.optimal_routing_forwarding(tos, msg, eth_type, ip_pkt.src, ip_pkt.dst)
                # ip
                if tos != 0 and tos not in self.tos_list_count and tos == self.tos_list_flag[-1]:
                    self.tos_list_count.append(tos)
                    self.tos_list_flag.append(tos + 1)
                    self.topology_module.arp_pkt_detect = False
                    self.send_flow_flag = True
                    # ==== handler tos == 192 ==== #
                    if tos + 1 == CF.DROP_TOS:
                        self.tos_list_count.append(tos + 1)
                        self.tos_list_flag.append(tos + 2)
                        self.count += 1
                    # set tos end #
                    if self.per_tos_end is None:
                        self.per_tos_end = len(self.topology_module.mapping_ip_mac.keys()) * (
                                len(self.topology_module.mapping_ip_mac.keys()) - 1)
                        if self.per_tos_end >= CF.DROP_TOS:
                            self.per_tos_end += 1
                    self.optimal_routing_forwarding(tos, msg, eth_type, ip_pkt.src, ip_pkt.dst)
                if tos == self.per_tos_end:
                    self.clear_tos_flag = True
                    self.send_flow_flag = False
                    return
                if self.clear_tos_flag:
                    del self.tos_list_count[:]
                    del self.tos_list_flag[:]
                    self.tos_list_flag = [1]
                    self.clear_tos_flag = False
                    self.send_flow_flag = False
                    self.count = 0
                    print("\033[33;1m clear %s tos list and prepare next iperf\033[0m" % self.per_tos_end)
                    print("\033[32;1m pause flow table distribution record\033[0m")
                    return
                return
