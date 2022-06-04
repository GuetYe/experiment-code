# -*- encoding: utf-8 -*-
"""
@File : traffic_matrix_measure.py 
@Author : hlq
@Modify Time : 4/12/22 12:07 AM
@Descipe: None
@Version : 1.0 
"""

import time
import networkx as nx
from ryu import cfg
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.base.app_manager import lookup_service_brick
from ryu.controller.handler import CONFIG_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ipv4
from ryu.lib.packet import arp
from ryu.lib import hub
import configuration as CF
from utils_file import SaveInfosToJson

import network_discover
import network_monitor
import network_delay
import network_manager

CONF = cfg.CONF

class TrafficMatrix(app_manager.RyuApp):
    """
        TrafficMatrix(Flow matrix monitoring mechanism)
            This class is used to monitor the network traffic matrix.
    """

    class_module = {
        "topology_module": network_discover.TopoDiscover,
        "monitor_module": network_monitor.MonitorDetection,
        "delay_module": network_delay.DelayMeasure,
        "management_module": network_manager.ManagementPlane
    }
    _CONTEXTS = class_module

    WEIGHT_MODEL = {'delay': 'delay', 'bw': "bandwidth"}

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]


    def __init__(self, *args, **kwargs):
        super(TrafficMatrix, self).__init__(*args, **kwargs)
        self.name = 'measure_module'
        self.topology_module = lookup_service_brick("topology_module")
        self.monitor_module = lookup_service_brick("status_module")
        self.delay_module = lookup_service_brick("delay_module")
        self.management_module = lookup_service_brick("management_module")
        self.candidate_path = {}
        self.count = 0
        # self.save_obj = None                      # save object
        # self.take_obj = None                      # take object
        self.json_obj = SaveInfosToJson()
        self.path_obj = self.json_obj.make_path
        self.per_tos_end = None                   # flag of the end of each flow
        self.tos_list_count = []                  # used identify tos
        self.tos_list_flag = [1]                  # used identify tos
        self.clear_tos_flag = False               # flag of clear tos_list_count and tos_list_flag
        self.send_flow_flag = False
        self.candidate_spawn = hub.spawn(self._discover_candidate_paths)
        self.link_spawn = hub.spawn(self._detect_traffic_matrix)

    def _discover_candidate_paths(self):
        """
            Discover candidate path
        :return:
        """
        i = 3
        while True:
            try:
                if self.topology_module is None:  # load topology_module
                    self.topology_module = lookup_service_brick("topology_module")
                if self.topology_module is not None:
                    if self.topology_module.arp_pkt_send:               # send arp pkt and start find candidate
                        hub.sleep(CF.DISCOVERY_CANDIDATE_PERIOD * i)    # wait DISCOVERY_CANDIDATE_PERIOD for first time
                        i = i-1 if i > 0 else 0                         # <=0
                        topo_data = self.path_obj.get_topo_data()
                        # the current candidate path exists. end the current collaboration. #
                        if topo_data is not None:
                            self.candidate_path = topo_data
                            print("candidate path has been exist! loading finished len %s!!" % self.get_dict_length(self.candidate_path))
                            break
                        # the current candidate path is not exists. find the candidate path by the current collaboration. #
                        total_len, nodes_len = self.get_candidate_paths()
                        cur_len = self.get_dict_length(self.candidate_path)
                        if total_len == cur_len:
                            print("\033[34;1m Compelete candidate path acquisition [len:%s] !!! \033[0m" % cur_len)
                            # self.save_obj.save_topo_info(self.candidate_path)      # save candidate paths
                            self.json_obj.save_all_path_infos(self.candidate_path) # save candidate paths
                            break
                hub.sleep(CF.DISCOVERY_CANDIDATE_PERIOD)
            except:
                hub.sleep(CF.DISCOVERY_CANDIDATE_PERIOD)

    def _detect_traffic_matrix(self):
        """
            Periodic detection traffic matrix and save it
        :return:
        """
        while True:
            if self.candidate_path and not self.topology_module.arp_pkt_detect and self.send_flow_flag:
                if self.management_module is None:
                    self.management_module = lookup_service_brick("management_module")
                link_infos = {}
                for src in self.candidate_path.keys():
                    link_infos.setdefault(src, {})
                    for dst in self.candidate_path.keys():
                        if src == dst: continue
                        # obtain the link information matrix[errors, drops, loss, delay ...] of the current path.
                        link_infos[src][dst] = self.management_module.get_traffic_matrix(self.candidate_path[src][dst])
                print("\n == == length of current traffic matrix %s == ==" % self.get_dict_length(link_infos))
                self.json_obj.save_all_metrics_infos(link_infos)
            time.sleep(CF.DETECT_TRAFFIC_MATRIX_PERIOD)

    def get_candidate_paths(self):
        """
            Get candidate paths to process DRL algo
        :return:
        """
        nodes = [dpid[0] for dpid in self.topology_module.mapping_ip_mac.keys()]                    # get current host's number
        for src in nodes:
            self.candidate_path.setdefault(src, {})
            for dst in nodes:
                if src == dst: continue
                self.candidate_path[src].setdefault(dst, [])
                # src->dst meet the current candidate path requirements. #
                if len(self.candidate_path[src][dst]) == CONF.k_paths:                              # meet
                    continue
                else:                                                                               # no meet
                    del self.candidate_path[src][dst][:]
                try:
                    generator = nx.shortest_simple_paths(self.topology_module.graph, source=src,
                                                         target=dst, weight="delay")               # use networkx to find candidate paths
                    src_dst_all_path = list(generator)
                    src_dst_all_path.sort(key=lambda x: len(x))                                     # sorted by path's len
                    if len(src_dst_all_path) >= CONF.k_paths:
                        self.candidate_path[src][dst].extend(src_dst_all_path[:CONF.k_paths])
                        print("src", src, "dst", dst, "has done!!!")
                    else:
                        print("src", src, "dst", dst, "non conformance xxx")
                except:
                    continue
        return len(nodes) * (len(nodes) - 1) * CONF.k_paths, len(nodes)

    def get_dict_length(self, dict_data):
        """
            Check the data' len of the current dict_data
        :param dict_data:
        :return:
        """
        dict_len = 0
        for src in dict_data.keys():
            for dst in dict_data[src].keys():
                dict_len += len(dict_data[src][dst])
        return dict_len

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
            Send flow by switch.
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
            Build packetOut pkt of forwarding.
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
            Send packetOut pkt by switch.
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
            Flood ARP packet to the access port which has no record of host.
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
            Get Current forwarding path.
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
                try:
                    path = nx.dijkstra_path(self.topology_module.graph, src_sw, dst_sw, weight="weight")
                except:
                    return
                if tos != 0:
                    if src_sw == dst_sw:
                        path = [src_sw]
                    else:
                        try:
                            path = nx.dijkstra_path(self.topology_module.graph, src_sw, dst_sw, weight="weight")
                        except:
                            print("no link path src : %s dst : %s" % (src_sw, dst_sw))
                            return
                flow_info = (eth_type, ip_src, ip_dst, in_port)
                self.install_flow(tos,  self.topology_module.datapaths, path,
                                  flow_info, msg.buffer_id, msg.data)
        return

    def arp_forwarding(self, msg, src_ip, dst_ip):
        """
            send ARP packet to the destination host, if the dst host record is existed,
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
