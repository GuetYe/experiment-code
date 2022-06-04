# _*_ coding:utf-8_*_
# Copyright (C) 2016 Li Cheng at Beijing University of Posts
# and Telecommunications. www.muzixing.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# conding=utf-8
import random
import copy
import threading
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
from ryu.lib import hub
from utils_file import *
from drl_tp.drlpr import DRL
from thread import start_new_thread, get_ident, LockType
CONF = cfg.CONF


class ShortestForwarding(app_manager.RyuApp):
    """
        ShortestForwarding is a Ryu app for forwarding packets in shortest
        path.
        The shortest path computation is done by module network awareness,
        network monitor and network delay detector.
    """

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    # _CONTEXTS = {
    #     "network_awareness": network_awareness.NetworkAwareness,
    #     "network_monitor": network_monitor.NetworkMonitor,
    #     "network_delay_detector": network_delay_detector.NetworkDelayDetector}

    WEIGHT_MODEL = {'hop': 'weight', 'delay': "delay", "bw": "bw", "db": "db", "hdb": "hdb", "sum": "sum"}

    def __init__(self, *args, **kwargs):
        super(ShortestForwarding, self).__init__(*args, **kwargs)
        self.name = 'shortest_forwarding'
        self.awareness = lookup_service_brick("awareness")
        self.monitor = lookup_service_brick("monitor")
        self.delay = lookup_service_brick("delay_detector")
        self.datapaths = {}
        self.weight = self.WEIGHT_MODEL[CONF.weight]
        self.best_paths = None
        self.tos_list = [0]                                                 # tos list
        self.tos_count = 0
        self.pre_tos = 0
        self.same_tos_flag = False
        self.all_metric_infos = {}                                         # save all metrics infos
        self.all_alternative_paths = None                                  # save all alternative paths
        self.drl_path = {}                                                 # drl path
        self.best_dijskra_path = {}
        self.drl_path_metric_infos = {}                                    #
        self.dijskra_path_metric_infos = {}
        self.ospf_path_metric_infos = {}
        # --- evaluating indicator --- #
        self.drl_link_rate = {}
        self.drl_link_delay = {}
        self.drl_link_loss = {}
        self.drl_link_used_bw = {}
        self.dijskra_link_rate = {}
        self.dijskra_link_delay = {}
        self.dijskra_link_loss = {}
        self.dijskra_link_used_bw = {}
        self.ospf_link_rate = {}
        self.ospf_link_delay = {}
        self.ospf_link_loss = {}
        self.ospf_link_used_bw = {}


        self.shortest_paths = None
        # self.get_drl_path = False
        self.get_drl_path = True
        if self.get_drl_path:
            self.drlpr = DRL()
        self.link_rate = {}                # save link_rate
        # self.path_obj = MakePath()
        self.json_obj = SaveInfosToJson()
        self.path_obj = self.json_obj.make_path
        # --- hub spawn --- #
        # self.detetct_metric_thread = hub.spawn(self._detect_all_metric)
        # self.ospf_forwarding_path = hub.spawn(self._ospf_forwarding)
        # self.dijskra_forwarding_path = hub.spawn(self._dijskra_forwarding)
        # self.drl_forwarding_path = hub.spawn(self._drl_forwarding)
        self.r_lock = threading.Lock()
        self.lock_type = LockType()
        # threading.Thread(target=self._detect_all_metric).start()
        # threading.Thread(target=self._ospf_forwarding).start()
        # threading.Thread(target=self._dijskra_forwarding).start()
        # threading.Thread(target=self._drl_forwarding).start()

        start_new_thread(self._detect_all_metric, )
        start_new_thread(self._ospf_forwarding, )
        start_new_thread(self._dijskra_forwarding, )
        start_new_thread(self._drl_forwarding, )

    def set_weight_mode(self, weight):
        """
            set weight mode of path calculating.
        """
        self.weight = weight
        if self.weight == self.WEIGHT_MODEL['hop']:
            self.awareness.get_shortest_paths(weight=self.weight)
        return True


    # --------------------- Based on drl to search rounting ------------------------- #
    def _detect_all_metric(self):
        """
            Detect all metric infos
        :return:
        """
        while True:
            if self.all_alternative_paths is None:
                if os.path.exists("topos" + "/" + str(time.strftime("%Y%m%d", time.localtime(time.time())))):
                    print("loading topo's infos ... ... ... ...")
                    with open("topos" + "/" + str(time.strftime("%Y%m%d", time.localtime(time.time()))) + "/" + "topo.json", "r") as json_file:  # get metrics/20210929/metric_1.json
                        all_path_infos = json.load(json_file)
                        all_path_infos = ast.literal_eval(json.dumps(all_path_infos))
                    self.all_alternative_paths = all_path_infos
                    self.shortest_paths = all_path_infos
                    # # test mode: link rate
                    if self.get_drl_path:
                        for link in setting.LINK_INFOS.keys():
                            self.drl_link_rate.setdefault(str(link), 0.)
                            self.drl_link_delay.setdefault(str(link), 0.)
                            self.drl_link_loss.setdefault(str(link), 0.)
                            self.drl_link_used_bw.setdefault(str(link), 0.)
                            self.dijskra_link_rate.setdefault(str(link), 0.)
                            self.dijskra_link_delay.setdefault(str(link), 0.)
                            self.dijskra_link_loss.setdefault(str(link), 0.)
                            self.dijskra_link_used_bw.setdefault(str(link), 0.)
                            self.ospf_link_rate.setdefault(str(link), 0.)
                            self.ospf_link_delay.setdefault(str(link), 0.)
                            self.ospf_link_loss.setdefault(str(link), 0.)
                            self.ospf_link_used_bw.setdefault(str(link), 0.)

                        # for src in self.all_alternative_paths.keys():
                        #     self.drl_link_delay.setdefault(src, {})
                        #     self.dijskra_link_delay.setdefault(src, {})
                        #     self.ospf_link_delay.setdefault(src, {})
                        #     for dst in self.all_alternative_paths[src].keys():
                        #         self.drl_link_delay[src][dst] = 0.
                        #         self.dijskra_link_delay[src][dst] = 0.
                        #         self.ospf_link_delay[src][dst] = 0.
                    # hub.sleep(setting.MONITOR_PERIOD * 5)
                    time.sleep(setting.MONITOR_PERIOD * 5)
            #         print("end ....")
            # for key, value in self.delay.link_delay.items():
            #     print(key, value)
            # --- get metric infos --- #
            try:
                for src in self.all_alternative_paths.keys():
                    # print(src)
                    for dst, paths in self.all_alternative_paths[src].items():
                        self.get_metric_infos(type_dict=self.all_metric_infos, src_sw=int(src), dst_sw=int(dst),
                                              graph=self.awareness.graph, paths=paths)

                if self.check_metric_is_format(self.all_metric_infos) == setting.NODES * (setting.NODES - 1):
                    # used agent to get drl path, test
                    if self.get_drl_path:
                        # self.r_lock.acquire()
                        self.lock_type.acquire()
                        self.drl_path  = self.drlpr.test_model(self.all_metric_infos)
                        # self.r_lock.release()
                        self.lock_type.release()
                    # train
                    else:
                        self.json_obj.save_all_metrics_infos(self.all_metric_infos, type="METRIC")
                        self.all_metric_infos.clear()
                # print("all_metric_infos's len is %s" % (inner_metric_len))
                # --- clear corresponding items --- #
                # self.all_metric_infos.clear()
            except Exception as e:
                self.all_metric_infos.clear()
            # hub.sleep(setting.MONITOR_PERIOD + 1)
            print("\033[36;1m detect [%s] done !!!\033[0m" % get_ident())
            hub.sleep(setting.MONITOR_PERIOD + 0.05)


    def _dijskra_forwarding(self):
        """
            Used dijskra to forwarding path
        :return:
        """
        while True:
            if self.drl_path:
                # self.r_lock.acquire()
                best_dijskra_path = {}
                for src in self.all_alternative_paths.keys():
                    for dst in self.all_alternative_paths[src].keys():
                        try:
                            # dijskra_path = [self.awareness.get_dijskra_path(int(src), int(dst))]
                            dijskra_path = self.awareness.k_shortest_paths(self.awareness.graph, int(src), int(dst))
                        except Exception as e:
                            self.logger.info("\033[34;1m src: %s dst: %s None find dijskra path\033[0m" % (src, dst))
                            dijskra_path = [random.choice(self.all_alternative_paths[src][dst])]
                            # if dijskra_path is None or dijskra_path == " ":
                            #     self.logger.info("src: %s dst: %s None find dijskra path" % (src, dst))
                            #     dijskra_path = [random.choice(self.all_alternative_paths[src][dst])]
                        # print("dijskra_path", dijskra_path)
                        best_dijskra_path.setdefault(src, {})
                        best_dijskra_path[src][dst] = dijskra_path
                        self.get_metric_infos(type_dict=self.dijskra_path_metric_infos, src_sw=int(src), dst_sw=int(dst),
                                              graph=self.awareness.graph, paths=dijskra_path)
                        self.get_link_throughout(type_dict=self.dijskra_link_rate, paths=dijskra_path)
                        self.get_link_delay(type_dict=self.dijskra_link_delay, paths=dijskra_path)
                        self.get_link_loss(type_dict=self.dijskra_link_loss, paths=dijskra_path)
                        self.get_link_use_bandwidth(type_dict=self.dijskra_link_used_bw, paths=dijskra_path)
                # can save dijskra's info
                if self.check_metric_is_format(best_dijskra_path) == setting.NODES * (setting.NODES - 1):
                    self.json_obj.save_all_path_infos(best_dijskra_path, type="DIJSKRA_PATH")                        # save d
                    self.json_obj.save_all_metrics_infos(self.dijskra_path_metric_infos, type="DIJSKRA_METRIC_PATH")
                    self.json_obj.save_all_metrics_infos(self.dijskra_link_rate, type="DIJSKRA_THROUGHTPUT_PATH")
                    self.json_obj.save_all_metrics_infos(self.dijskra_link_delay, type="DIJSKRA_DELAY_PATH")
                    self.json_obj.save_all_metrics_infos(self.dijskra_link_loss, type="DIJSKRA_LOSS_PATH")
                    self.json_obj.save_all_metrics_infos(self.dijskra_link_used_bw, type="DIJSKRA_USE_BW_PATH")
                self.dijskra_path_metric_infos.clear()
                self.set_dict_to_zero(self.dijskra_link_rate)
                self.set_dict_to_zero(self.dijskra_link_delay)
                # self.set_all_dict_to_zero(self.dijskra_link_delay)
                self.set_dict_to_zero(self.dijskra_link_loss)
                self.set_dict_to_zero(self.dijskra_link_used_bw)
                print("\033[35;1m dijskra [%s] done !!!\033[0m" % get_ident())
            # self.r_lock.release()
            hub.sleep(setting.MONITOR_PERIOD + 0.15)



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

    def set_dict_to_zero(self, type_dict):
        """

        :param type_dict:
        :return:
        """
        for key in type_dict.keys():
            type_dict[key] = 0.

    def set_all_dict_to_zero(self, type_dict):
        for src in type_dict.keys():
            for dst in type_dict[src].keys():
                type_dict[src][dst] = 0.0



    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        """
            Collect datapath information.
        """
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if not datapath.id in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def add_flow(self, dp, p, match, actions, idle_timeout=0, hard_timeout=0):
        """
            Send a flow entry to datapath.
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

    def _build_packet_out(self, datapath, buffer_id, src_port, dst_port, data):
        """
            Build packet out object.
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
            Send packet out packet to assigned datapath.
        """
        out = self._build_packet_out(datapath, buffer_id,
                                     src_port, dst_port, data)
        if out:
            datapath.send_msg(out)

    def get_port(self, dst_ip, access_table):
        """
            Get access port if dst host.
            access_table: {(sw,port) :(ip, mac)}
        """
        if access_table:
            if isinstance(access_table.values()[0], tuple):
                for key in access_table.keys():
                    if dst_ip == access_table[key][0]:
                        dst_port = key[1]
                        return dst_port
        return None

    def get_port_pair_from_link(self, link_to_port, src_dpid, dst_dpid):
        """
            Get port pair of link, so that controller can install flow entry.
        """
        if (src_dpid, dst_dpid) in link_to_port:
            return link_to_port[(src_dpid, dst_dpid)]
        else:
            self.logger.info("dpid:%s->dpid:%s is not in links" % (
                src_dpid, dst_dpid))
            return None

    def flood(self, msg):
        """
            Flood ARP packet to the access port
            which has no record of host.
        """
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        for dpid in self.awareness.access_ports:
            for port in self.awareness.access_ports[dpid]:
                if (dpid, port) not in self.awareness.access_table.keys():
                    datapath = self.datapaths[dpid]
                    out = self._build_packet_out(
                        datapath, ofproto.OFP_NO_BUFFER,
                        ofproto.OFPP_CONTROLLER, port, msg.data)
                    datapath.send_msg(out)
        self.logger.debug("Flooding msg")

    def arp_forwarding(self, msg, src_ip, dst_ip):
        """ Send ARP packet to the destination host,
            if the dst host record is existed,
            else, flow it to the unknow access port.
        """
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        result = self.awareness.get_host_location(dst_ip)
        if result:  # host record in access table.
            datapath_dst, out_port = result[0], result[1]
            datapath = self.datapaths[datapath_dst]
            out = self._build_packet_out(datapath, ofproto.OFP_NO_BUFFER,
                                         ofproto.OFPP_CONTROLLER,
                                         out_port, msg.data)
            datapath.send_msg(out)
            self.logger.debug("Reply ARP to knew host")

        else:
            self.flood(msg)

    def get_min_dl_of_links(self, graph, path, min_dl=0.):
        """Getting total link delay"""
        _len = len(path)
        # --- the link node length is greater than 1 --- #
        if _len > 1:
            minimal_delay = min_dl
            for i in xrange(_len - 1):
                pre, curr = path[i], path[i + 1]
                if 'lldpdelay' in graph[pre][curr]:
                    dl = graph[pre][curr]['lldpdelay']
                    minimal_delay = minimal_delay + dl
                    # print("-------minimal_delay:%s" % minimal_delay)
                else:
                    continue
            return minimal_delay
        return min_dl


    def get_total_delay_of_path(self, graph, path, total_delay=0.):
        """
        Get total delay of a path
        :param graph:
        :param path:
        :param total_delay:
        :return:
        """
        _len = len(path)
        min_delay = 0.
        if _len > 1:
            for i in range(_len - 1):
                pre_sw, curr_sw = int(path[i]), int(path[i + 1])
                # try:
                #     total_delay += graph[pre_sw][curr_sw]['delay']
                # except Exception as e:
                #     continue
                if 'delay' in graph[pre_sw][curr_sw]:
                    total_delay += graph[pre_sw][curr_sw]['delay']
                else:
                    continue
            # make sure the delay is not -inf
            if min_delay > total_delay or 9999999999 < total_delay:
                total_delay = min_delay
            return total_delay

        return total_delay


    def get_best_path_by_delay(self, tos, ip_src, ip_dst, graph, paths):
        """
            Get best path used in Ping packet
        :param tos:
        :param ip_src:
        :param ip_dst:
        :param graph:
        :param paths:
        :return:
        """
        infos = " "
        # _graph = copy.deepcopy(graph)
        # _paths = copy.deepcopy(paths)
        best_path, best_path_bw, best_path_delay = None, None, float("inf")
        for path in paths:
            total_path_delay = self.get_total_delay_of_path(graph, path, 0.)
            if best_path_delay > total_path_delay:
                best_path = path
                best_path_delay = total_path_delay
                best_path_bw = self.monitor.get_min_bw_of_links(graph, path, setting.MAX_CAPACITY)
        # if tos != 0:
        #     src, dst = best_path[0], best_path[-1]
        #     self.dijskra_path_metric_infos.setdefault()
        #     self.get_link_packet_drop()
        # self.logger.info("\033[33;1m--------------delay:%s\033[0m" % best_path_delay)
        # self.logger.info("\033[33;1m----------bandwidth:%s\033[0m" % best_path_bw)
        # self.logger.info("\033[36;1m----------[PATH]%s<-->%s: %s\033[0m" % (ip_src, ip_dst, best_path))
        # infos += str(tos) + "\n" + str(best_path_delay) + "\n" + str(best_path_bw) + "\n" + str(
        #     "%s<--->%s : %s") % (ip_src, ip_dst, best_path) + "\n"
        # self.operator_file.check_file("delay_bw", setting.CURRENT_TIME, infos)

        # print("best_path %s" % best_path)

        return [best_path], best_path_bw, best_path_delay

    def get_sw(self, dpid, in_port, src, dst):
        """
            Get pair of source and destination switches.
        """
        src_sw = dpid
        dst_sw = None

        src_location = self.awareness.get_host_location(src)
        if in_port in self.awareness.access_ports[dpid]:
            if (dpid, in_port) == src_location:
                src_sw = src_location[0]
            else:
                return None

        dst_location = self.awareness.get_host_location(dst)
        if dst_location:
            dst_sw = dst_location[0]

        return src_sw, dst_sw



    def get_path(self, tos, src, dst, ip_src, ip_dst, weight):
        """
            Get shortest path from network awareness module.
            shortest_paths:type:dict
        {1: {1: [[1], [1]], 2: [[1, 2], [1, 3, 5, 2]], 3: [[1, 3], [1, 2, 5, 3]], 4: [[1, 4], [1, 2, 5, 4]], 5: [[1, 2, 5], [1, 3, 5],[1,4,5]]},
         2: {1: [[2, 1], [2, 5, 3, 1]], 2: [[2], [2]], 3: [[2, 1, 3], [2, 5, 3]], 4: [[2, 1, 4], [2, 5, 4]], 5: [[2, 5], [2, 1, 3, 5]]},
         3: {1: [[3, 1], [3, 5, 2, 1]], 2: [[3, 1, 2], [3, 5, 2]], 3: [[3], [3]], 4: [[3, 1, 4], [3, 5, 4]], 5: [[3, 5], [3, 1, 2, 5]]},
         4: {1: [[4, 1], [4, 5, 2, 1]], 2: [[4, 1, 2], [4, 5, 2]], 3: [[4, 1, 3], [4, 5, 3]], 4: [[4], [4]], 5: [[4, 5], [4, 1, 2, 5]]},
         5: {1: [[5, 2, 1], [5, 3, 1],[5,4,1]], 2: [[5, 2], [5, 3, 1, 2]], 3: [[5, 3], [5, 2, 1, 3]], 4: [[5, 4], [5, 2, 1, 4]], 5: [[5], [5]]}}

        """
        # paths:type:list [[1]],[[5,2,1]]
        # shortest_paths = self.awareness.shortest_paths
        graph = self.awareness.graph
        # print("graph", graph.nodes())
        # print("------", shortest_paths)
        temp_paths = []

        if weight == self.WEIGHT_MODEL['hop']:
            paths = self.shortest_paths[str(src)][str(dst)][0]
            temp_paths.append(paths)
            return temp_paths

        elif weight == self.WEIGHT_MODEL['delay']:
            # If paths existed, return it, else calculate it and save it.
            try:
                paths = self.shortest_paths[str(src)][str(dst)]
                temp_paths.append(paths[0])
                return temp_paths
            except:
                paths = self.awareness.k_shortest_paths(graph, src, dst,
                                                        weight=weight)

                # shortest_paths.setdefault(src, {})
                # shortest_paths[src].setdefault(dst, paths)
                temp_paths.append(paths[0])
                return temp_paths

        elif weight == self.WEIGHT_MODEL['bw']:  # based on bw
            # Because all paths will be calculate
            # when call self.monitor.get_best_path_by_bw
            # So we just need to call it once in a period,
            # and then, we can get path directly.
            try:
                # if path is existed, return it.
                path = self.monitor.best_paths.get(str(src)).get(str(dst))
                return path
            except:
                # else, calculate it, and return.
                result = self.monitor.get_best_path_by_bw(graph,
                                                          self.shortest_paths)
                paths = result[1]
                best_path = paths.get(src).get(dst)

                return best_path

        elif weight == self.WEIGHT_MODEL["db"]:
            # delay_paths,bw_paths = None,None
            # --- Get the optimal delay path --- #
            if src == dst:
                if tos != 0:
                    return
                return [[src]]
            else:
                try:
                    # print(self.shortest_paths)
                    # self.awareness.shortest_paths
                    # paths = self.shortest_paths.get(str(src)).get(str(src))
                    paths = self.shortest_paths[str(src)][str(dst)]
                    # print(paths)
                except:
                    print("src :%s --> dst :%s No path" % (src, dst))
                    paths = None
                if paths:
                    if tos != 0:
                        print("H-%s is streaming [%s] H-%s " % (ip_src, tos, ip_dst))
                        return paths
                    # ping packet infos, need to send flow
                    best_path, best_bw, best_delay = self.get_best_path_by_delay(tos, ip_src, ip_dst, graph, paths)
                    return best_path
                return

        elif weight == self.WEIGHT_MODEL["hdb"]:
            # Get hop delay and bw
            pass


    # ===================================  DRL part ====================================== #
    def get_all_alternative_paths(self, src_sw, dst_sw, paths):
        """
            Get all alternative paths
        """
        self.all_alternative_paths.setdefault(src_sw, {})
        self.all_alternative_paths[src_sw].setdefault(dst_sw, None)
        self.all_alternative_paths[src_sw][dst_sw] = paths

    def get_link_packet_loss(self, paths):
        """
            Get packet loss
        """
        paths = copy.deepcopy(paths)
        packet_loss = []
        for path in paths:
            each_link_loss = 0.
            for i in range(len(path) - 1):
                src_sw, dst_sw = int(path[i]), int(path[i + 1])
                try:
                    src_port, dst_port = self.awareness.link_to_port[(src_sw, dst_sw)]
                    # get src bytes packets
                    src_sw_infos = self.monitor.port_stats[(src_sw, src_port)]

                    # src_port, dst_port = self.awareness.link_to_port[(src_sw, dst_sw)]
                    # # get src bytes packets
                    # src_sw_infos = self.monitor.port_stats[(src_sw, src_port)]
                    #
                    # src_tx_bytes = src_sw_infos[-1][0]
                    # src_tx_packets = src_sw_infos[-1][8]
                    # # get dst bytes packets
                    # dst_sw_infos = self.monitor.port_stats[(dst_sw, dst_port)]
                    # dst_rx_byters = dst_sw_infos[-1][1]
                    # dst_rx_packets = dst_sw_infos[-1][9]
                    #
                    #
                    #
                    # # if len(src_sw_infos) < 2:
                    # #     continue
                    # tx_bytes = src_sw_infos[-1][0]
                    # rx_bytes = src_sw_infos[-2][1]
                    # # src_tx_packets = src_sw_infos[-1][8]
                    # # get dst bytes packets
                    # # dst_sw_infos = self.monitor.port_stats[(dst_sw, dst_port)]
                    # # dst_rx_byters = dst_sw_infos[-1][1]
                    # # dst_rx_packets = dst_sw_infos[-2][9]
                    # # loss = (src_tx_packets - dst_rx_packets) / src_tx_packets
                    # loss = (float(tx_bytes - rx_bytes)) / tx_bytes
                    # if loss < 0:
                    #     loss = abs(loss)
                    # # loss = (src_tx_packets - dst_rx_packets) / src_tx_packets
                    # # max_packets = max(src_tx_packets, dst_rx_packets)
                    # # min_packets = min(src_tx_packets, dst_rx_packets)
                    # # loss = (float(max_packets) - min_packets) / max_packets
                    # each_link_loss += loss

                    tx_bytes = src_sw_infos[-2][0]
                    rx_bytes = src_sw_infos[-1][1]


                    # src_tx_packets = src_sw_infos[-1][8]
                    # src_rx_
                    # # get dst bytes packets
                    # dst_sw_infos = self.monitor.port_stats[(dst_sw, dst_port)]
                    # dst_rx_byters = dst_sw_infos[-1][1]
                    # dst_rx_packets = dst_sw_infos[-2][9]
                    # # loss = (src_tx_packets - dst_rx_packets) / src_tx_packets
                    loss = (float(tx_bytes - rx_bytes)) / tx_bytes
                    # if loss < 0:
                    #     loss = abs(loss)
                    each_link_loss += loss
                    # print(src_tx_bytes)
                except Exception as e:
                    # print(src_sw, dst_sw)
                    continue
            packet_loss.append(each_link_loss)

        return packet_loss

    def get_link_packet_drop(self, paths):
        """
            Get paket drop
        """
        paths = copy.deepcopy(paths)
        packet_drop = []
        for path in paths:
            each_link_drop = 0.
            for i in range(len(path) - 1):
                src_sw, dst_sw = int(path[i]), int(path[i + 1])
                try:
                    src_port, dst_port = self.awareness.link_to_port[(src_sw, dst_sw)]
                    # get src bytes packets
                    src_sw_infos = self.monitor.port_stats[(src_sw, src_port)]
                    src_tx_drop = src_sw_infos[-1][6]
                    # get dst bytes packets
                    dst_sw_infos = self.monitor.port_stats[(dst_sw, dst_port)]
                    dst_rx_drop = dst_sw_infos[-1][7]
                    drop = abs(src_tx_drop - dst_rx_drop)
                    # drop = (src_tx_drop - dst_rx_drop) / src_tx_drop
                    each_link_drop += drop
                except Exception as e:
                    continue
            packet_drop.append(each_link_drop)

        return packet_drop

    def get_link_packet_errors(self, paths):
        """
        Get link_packet_errors
        """
        paths = copy.deepcopy(paths)
        packet_errors = []
        for path in paths:
            each_link_errors = 0.
            for i in range(len(path) - 1):
                src_sw, dst_sw = int(path[i]), int(path[i + 1])
                try:
                    src_port, dst_port = self.awareness.link_to_port[(src_sw, dst_sw)]
                    # get src bytes packets
                    src_sw_infos = self.monitor.port_stats[(src_sw, src_port)]
                    src_tx_errors = src_sw_infos[-1][5]
                    # get dst bytes packets
                    dst_sw_infos = self.monitor.port_stats[(dst_sw, dst_port)]
                    dst_rx_errors = dst_sw_infos[-1][2]
                    errors = abs((src_tx_errors - dst_rx_errors))
                    each_link_errors += errors
                except Exception as e:
                    continue
            packet_errors.append(each_link_errors)

        return packet_errors

    def get_link_used_bw(self, paths):
        """
        Get used_bw of links
        :return:
        """
        paths = copy.deepcopy(paths)
        used_bw = []
        for path in paths:
            each_link_used_bw = 0.
            for i in range(len(path) - 1):
                src_sw, dst_sw = int(path[i]), int(path[i + 1])
                try:
                    # get src port dst port
                    src_port, dst_port = self.awareness.link_to_port[(src_sw, dst_sw)]
                    # get src and dst used bw
                    # src_sw_infos = self.monitor.port_stats[(src_sw, src_port)]
                    # if len(src_sw_infos) < 2:
                    #     continue
                    # cur_tx_bytes = src_sw_infos[-1][0]
                    # pre_tx_bytes = src_sw_infos[-2][0]
                    # used_bw = (float(cur_tx_bytes * 8 - pre_tx_bytes * 8)) / setting.MONITOR_PERIOD
                    # if used_bw < 0:
                    #     used_bw = abs(used_bw)
                    used_bw1 = self.monitor.port_speed[(src_sw, src_port)][-1] * 8 / 10 ** 6
                    used_bw2 = self.monitor.port_speed[(dst_sw, dst_port)][-1] * 8 / 10 ** 6
                    # link_used_bw = (used_bw1 + used_bw2) / 2 / 1000
                    each_link_used_bw = max(each_link_used_bw, max(used_bw1, used_bw2))
                    # each_link_used_bw = used_bw
                except Exception as e:
                    continue
            used_bw.append(each_link_used_bw)
            # print(each_link_used_bw)
        return used_bw


    def get_link_throughout(self, type_dict, paths):
        """
            Compute link throughout
        :param paths:
        :return:
        """

        for path in paths:
            for i in range(len(path) - 1):
                try:
                    src_sw, dst_sw = int(path[i]), int(path[i + 1])
                    if (src_sw, dst_sw) in setting.LINK_INFOS.keys():
                        src_port, dst_port = self.awareness.link_to_port[(src_sw, dst_sw)]
                        sw_infos = self.monitor.port_stats[(src_sw, src_port)]
                        pre_tx_bytes = sw_infos[-2][0]
                        cur_tx_bytes = sw_infos[-1][0]
                        # tx_packets = sw_infos[-1][8]
                        link_bw = float("inf")
                        if 'bandwidth' in self.awareness.graph[src_sw][dst_sw]:
                            bw = self.awareness.graph[src_sw][dst_sw]['bandwidth']  # Mb/s
                            link_bw = min(bw * 10 ** 3, link_bw)
                        tx_total_bytes = pre_tx_bytes + cur_tx_bytes
                        link_bw = min(setting.LINK_INFOS[(src_sw, dst_sw)], link_bw)
                        # type_dict[str((src_sw, dst_sw))] += (tx_total_bytes * 8 / (link_bw * 10** 3 * (setting.MONITOR_PERIOD - 0.5)))
                        type_dict[str((src_sw, dst_sw))] += (
                                    tx_total_bytes * 8 / (link_bw * (setting.MONITOR_PERIOD - 0.5)))
                except Exception as e:
                    # print("link throughput : %s path :%s Error!" % (type(type), path))
                    continue





        # for path in paths:
        #     for i in range(len(path) - 1):
        #         try:
        #             src_sw, dst_sw = int(path[i]), int(path[i + 1])
        #             if (src_sw, dst_sw) in setting.LINK_INFOS.keys():
        #                 src_port, dst_port = self.awareness.link_to_port[(src_sw, dst_sw)]
        #                 # get src bytes packets
        #                 sw_infos = self.monitor.port_stats[(src_sw, src_port)]
        #                 tx_bytes = sw_infos[-1][0]
        #                 # tx_packets = sw_infos[-1][8]
        #                 link_bw = float("inf")
        #                 if 'bandwidth' in self.awareness.graph[src_sw][dst_sw]:
        #                     bw = self.awareness.graph[src_sw][dst_sw]['bandwidth']  # Mb/s
        #                     link_bw = min(bw * 10 ** 3, link_bw)
        #                 link_bw = min(setting.LINK_INFOS[(src_sw, dst_sw)], link_bw)
        #                 type_dict[str((src_sw, dst_sw))] += (tx_bytes * 8 / link_bw)
        #         except Exception as e:
        #             # print("link throughput : %s path :%s Error!" % (type(type), path))
        #             continue

    def get_link_delay(self, type_dict, paths):
        """
            Compute link throughout
        :param paths:
        :return:
        """
        for path in paths:
            # src, dst = path[0], path[-1]
            # try:
                # for src in type_dict.keys():
                #     for dst, value in type_dict[src].items():
                #         print(src, dst, value)
            # delay = self.get_total_delay_of_path(self.awareness.graph, path, 0.)
            # type_dict[str(path[0])][str(path[-1])] += delay
            # except Exception as e:
            #     print(path[0], path[-1], type(path[0]))
            for i in range(len(path) - 1):
                try:
                    src_sw, dst_sw = int(path[i]), int(path[i + 1])
                    if (src_sw, dst_sw) in setting.LINK_INFOS.keys():
                        # bw = setting.LINK_INFOS[(src_sw, dst_sw)]
                        # if 'bandwidth' in self.awareness.graph[src_sw][dst_sw]:
                        #     # bw = self.awareness.graph[src_sw][dst_sw]['bandwidth']
                        #     bw = min(bw, self.awareness.graph[src_sw][dst_sw]['bandwidth'])
                        # delay = max(0, self.awareness.graph[src_sw][dst_sw]['delay'])
                        delay = self.delay.link_delay[(src_sw, dst_sw)]
                        # if 'delay' in self.awareness.graph[src_sw][dst_sw]:
                        #     # print("------" , self.awareness.graph[src_sw][dst_sw])
                        #     delay = max(delay, self.awareness.graph[src_sw][dst_sw]['delay'])
                        # if delay < 0 or delay > 99999999999999:
                        #     delay = 0.
                        # type_dict[str((src_sw, dst_sw))] += self.delay.link_delay[(src_sw, dst_sw)]
                        type_dict[str((src_sw, dst_sw))] = max(type_dict[str((src_sw, dst_sw))], delay)
                except Exception as e:
                    # print("link delay: %s path :%s Error!" % (type(type), path) )
                    continue

    def get_link_loss(self, type_dict, paths):
        """

        :return:
        """
        for path in paths:
            for i in range(len(path) - 1):
                src_sw, dst_sw = int(path[i]), int(path[i + 1])
                try:
                    # src_port, dst_port = self.awareness.link_to_port[(src_sw, dst_sw)]
                    if (src_sw, dst_sw) in setting.LINK_INFOS.keys():
                        src_port, dst_port = self.awareness.link_to_port[(src_sw, dst_sw)]
                        # get src bytes packets
                        src_sw_infos = self.monitor.port_stats[(src_sw, src_port)]
                        dst_sw_infos = self.monitor.port_stats[(dst_sw, dst_port)]
                        src_tx_bytes = src_sw_infos[-1][0]
                        src_rx_bytes = src_sw_infos[-1][1]
                        src_tx_pkts = src_sw_infos[-1][8]
                        src_rx_pkts = src_sw_infos[-1][9]
                        dst_tx_bytes = dst_sw_infos[-1][0]
                        dst_rx_bytes = dst_sw_infos[-1][1]
                        dst_tx_pkts = dst_sw_infos[-1][8]
                        dst_rx_pkts = dst_sw_infos[-1][9]
                        # loss = (float(src_tx_bytes - src_rx_bytes)) / src_tx_bytes + \
                        #        (float(dst_tx_bytes - dst_rx_bytes)) / dst_tx_bytes
                        loss = (float(src_tx_pkts - src_rx_pkts)) / src_tx_pkts + \
                               (float(dst_tx_pkts - dst_rx_pkts)) / dst_tx_pkts
                        # loss = (float(src_tx_pkts - src_rx_pkts)) / src_tx_pkts
                        # if loss < 0:
                        #     loss = abs(loss)
                        type_dict[str((src_sw, dst_sw))] += loss
                    # print(src_tx_bytes)
                except Exception as e:
                    # print("link loss: %s path :%s Error!" % (type(type), path) )
                    continue

    # def get_link_loss(self, type_dict, paths):
    #     """
    #
    #     :return:
    #     """
    #     for path in paths:
    #         for i in range(len(path) - 1):
    #             src_sw, dst_sw = int(path[i]), int(path[i + 1])
    #             try:
    #                 # src_port, dst_port = self.awareness.link_to_port[(src_sw, dst_sw)]
    #                 if (src_sw, dst_sw) in setting.LINK_INFOS.keys():
    #                     src_port, dst_port = self.awareness.link_to_port[(src_sw, dst_sw)]
    #                     # get src bytes packets
    #                     src_sw_infos = self.monitor.port_stats[(src_sw, src_port)]
    #                     tx_bytes = src_sw_infos[-2][0]
    #                     rx_bytes = src_sw_infos[-1][1]
    #                     loss = (float(tx_bytes - rx_bytes)) / tx_bytes
    #                     if loss < 0:
    #                         loss = abs(loss)
    #                     type_dict[str((src_sw, dst_sw))] += loss
    #                 # print(src_tx_bytes)
    #             except Exception as e:
    #                 # print("link loss: %s path :%s Error!" % (type(type), path) )
    #                 continue


    def get_link_use_bandwidth(self, type_dict, paths):
        for path in paths:
            for i in range(len(path) - 1):
                src_sw, dst_sw = int(path[i]), int(path[i + 1])
                try:
                    # src_port, dst_port = self.awareness.link_to_port[(src_sw, dst_sw)]
                    if (src_sw, dst_sw) in setting.LINK_INFOS.keys():
                        src_port, dst_port = self.awareness.link_to_port[(src_sw, dst_sw)]
                        # get src bytes packets
                        src_sw_infos = self.monitor.port_stats[(src_sw, src_port)]
                        dst_sw_infos = self.monitor.port_stats[(dst_sw, dst_port)]
                        # if len(src_sw_infos) >= 2:
                        cur_src_rx_bytes = src_sw_infos[-1][1]
                        pre_src_rx_bytes = src_sw_infos[-2][1]
                        cur_dst_rx_bytes = src_sw_infos[-1][1]
                        pre_dst_rx_bytes = src_sw_infos[-2][1]
                        used_src_bw = float(cur_src_rx_bytes * 8 / 10 ** 3 - pre_src_rx_bytes * 8 / 10 ** 3) / (setting.MONITOR_PERIOD - 0.5)
                        used_dst_bw = float(cur_dst_rx_bytes * 8 / 10 ** 3 - pre_dst_rx_bytes * 8 / 10 ** 3) / (setting.MONITOR_PERIOD - 0.5)
                        # loss = (float(cur_rx_bytes - pre_rx_bytes)) / tx_bytes
                        used_bw = max(used_src_bw, used_dst_bw)
                        type_dict[str((src_sw, dst_sw))] = max(type_dict[str((src_sw, dst_sw))], used_bw)
                    # print(src_tx_bytes)
                except Exception as e:
                    # print("link loss: %s path :%s Error!" % (type(type), path) )
                    continue



    def get_metric_infos(self, type_dict, src_sw, dst_sw, graph, paths):
            """
                Get bw delay loss ...
            :return
            """

            if paths is None:
                return
            # try:
            type_dict.setdefault(src_sw, {})
            type_dict[src_sw].setdefault(dst_sw, {})
            type_dict[src_sw][dst_sw].setdefault("free_bw", [])
            type_dict[src_sw][dst_sw].setdefault("delay", [])
            type_dict[src_sw][dst_sw].setdefault("used_bw", [])
            type_dict[src_sw][dst_sw].setdefault("packet_loss", [])
            type_dict[src_sw][dst_sw].setdefault("packet_drop", [])
            type_dict[src_sw][dst_sw].setdefault("packet_errors", [])
            # Obatain metric infos
            for path in paths:
                total_path_delay = self.get_total_delay_of_path(graph, path, 0.)
                path_bw = self.monitor.get_min_bw_of_links(graph, path, setting.MAX_CAPACITY)
                type_dict[src_sw][dst_sw]["delay"].append(total_path_delay)
                type_dict[src_sw][dst_sw]["free_bw"].append(path_bw)
            type_dict[src_sw][dst_sw]["used_bw"].extend(self.get_link_used_bw(paths))
            # print("use")
            type_dict[src_sw][dst_sw]["packet_loss"].extend(self.get_link_packet_loss(paths))
            # print("loss")
            type_dict[src_sw][dst_sw]["packet_drop"].extend(self.get_link_packet_drop(paths))
            # print("drop")
            type_dict[src_sw][dst_sw]["packet_errors"].extend(self.get_link_packet_errors(paths))
                # print("error")
            # except Exception as e:
            #     pass

    # def
    #         if self.get_drl_path:
    #             self.get_link_throughout(type_dict, paths)



    # ===============================  Send flow part ==================================== #
    def send_flow_mod(self, tos, datapath, flow_info, src_port, dst_port):
        """
            Build flow entry, and send it to datapath.
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

    def install_flow(self, tos, datapaths, link_to_port, access_table, path,
                     flow_info, buffer_id, data=None):
        '''
            Install flow entires for roundtrip: go and back.
            @parameter: path=[dpid1, dpid2...]
                        flow_info=(eth_type, src_ip, dst_ip, in_port)
        '''
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
                port = self.get_port_pair_from_link(link_to_port,
                                                    path[i - 1], path[i])
                port_next = self.get_port_pair_from_link(link_to_port,
                                                         path[i], path[i + 1])
                if port and port_next:
                    src_port, dst_port = port[1], port_next[0]
                    datapath = datapaths[path[i]]
                    self.send_flow_mod(tos, datapath, flow_info, src_port, dst_port)
                    self.send_flow_mod(tos, datapath, back_info, dst_port, src_port)
                    self.logger.debug("inter_link flow install")
        if len(path) > 1:
            # the last flow entry: tor -> host
            port_pair = self.get_port_pair_from_link(link_to_port,
                                                     path[-2], path[-1])
            if port_pair is None:
                self.logger.info("Port is not found")
                return
            src_port = port_pair[1]

            dst_port = self.get_port(flow_info[2], access_table)
            if dst_port is None:
                self.logger.info("Last port is not found.")
                return

            last_dp = datapaths[path[-1]]
            self.send_flow_mod(tos, last_dp, flow_info, src_port, dst_port)
            self.send_flow_mod(tos, last_dp, back_info, dst_port, src_port)

            # the first flow entry
            port_pair = self.get_port_pair_from_link(link_to_port,
                                                     path[0], path[1])
            if port_pair is None:
                self.logger.info("Port not found in first hop.")
                return
            out_port = port_pair[0]
            self.send_flow_mod(tos, first_dp, flow_info, in_port, out_port)
            self.send_flow_mod(tos, first_dp, back_info, out_port, in_port)
            self.send_packet_out(first_dp, buffer_id, in_port, out_port, data)

        # src and dst on the same datapath
        else:
            out_port = self.get_port(flow_info[2], access_table)
            if out_port is None:
                self.logger.info("Out_port is None in same dp")
                return
            self.send_flow_mod(tos, first_dp, flow_info, in_port, out_port)
            self.send_flow_mod(tos, first_dp, back_info, out_port, in_port)
            self.send_packet_out(first_dp, buffer_id, in_port, out_port, data)

    def shortest_forwarding(self, tos, msg, eth_type, ip_src, ip_dst):
        """
            To calculate shortest forwarding path and install them into datapaths.
        """
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        result = self.get_sw(datapath.id, in_port, ip_src, ip_dst)
        self.result = result
        if result:
            src_sw, dst_sw = result[0], result[1]
            if dst_sw:
                # Path has already calculated, just get it.

                if self.drl_path:
                    path = self.drl_path[str(src_sw)][str(dst_sw)]
                    print("drl_path", path, "----------")
                    # for path in paths:
                        # path = self.get_path(src_sw, dst_sw, weight=self.weight)
                        # self.logger.info("--------------------------[PATH]%s<-->%s: %s" % (ip_src, ip_dst, path))
                        # print("-------------------------------------------------------------")
                    flow_info = (eth_type, ip_src, ip_dst, in_port)
                    # install flow entries to datapath along side the path.
                    self.install_flow(tos, self.datapaths,
                                      self.awareness.link_to_port,
                                      self.awareness.access_table, path,
                                      flow_info, msg.buffer_id, msg.data)
                return
                # paths = self.get_path(tos, src_sw, dst_sw, ip_src, ip_dst, weight=self.weight)
                # # if paths and tos == 0:
                # if paths:
                #     for path in paths:
                #         # path = self.get_path(src_sw, dst_sw, weight=self.weight)
                #         # self.logger.info("--------------------------[PATH]%s<-->%s: %s" % (ip_src, ip_dst, path))
                #         # print("-------------------------------------------------------------")
                #         flow_info = (eth_type, ip_src, ip_dst, in_port)
                #         # install flow entries to datapath along side the path.
                #         self.install_flow(tos, self.datapaths,
                #                           self.awareness.link_to_port,
                #                           self.awareness.access_table, path,
                #                           flow_info, msg.buffer_id, msg.data)
        return


    # =============================  Listen Event ============================== #
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        '''
            In packet_in handler, we need to learn access_table by ARP.
            Therefore, the first packet from UNKOWN host MUST be ARP.
        '''
        msg = ev.msg
        datapath = msg.datapath
        in_port = msg.match['in_port']
        pkt = packet.Packet(msg.data)
        arp_pkt = pkt.get_protocol(arp.arp)
        ip_pkt = pkt.get_protocol(ipv4.ipv4)

        if isinstance(arp_pkt, arp.arp):
            # self.logger.info("Send ARP Packet...")
            self.logger.debug("ARP processing")
            self.arp_forwarding(msg, arp_pkt.src_ip, arp_pkt.dst_ip)

        if isinstance(ip_pkt, ipv4.ipv4):
            # self.logger.info("Send IPV4 Packet...")
            self.logger.debug("IPV4 processing")
            tos = ip_pkt.tos            # int
            # tos_flag = 0
            # if tos != 0:
            #     tos_flag = tos + self.tos_count * SaveInfosToJson.each_total_tos
            #     # don't let same tos into tos_list
            #     if tos == SaveInfosToJson.each_total_tos and self.same_tos_flag:
            #         return
            #     # only one
            #     if tos == SaveInfosToJson.each_total_tos and not self.same_tos_flag:
            #         self.tos_count += 1
            #         self.pre_tos = tos_flag
            #         self.same_tos_flag = True
            if len(pkt.get_protocols(ethernet.ethernet)):
                # ping packets
                # if tos == 192: return
                # # if tos == 0 or tos not in self.tos_list:
                # if tos == 2 ** 8 - 1:
                #     del self.tos_list[:]
                #     return
                # if tos != 0 and tos not in self.tos_list :
                #     self.tos_list.append(tos)
                #     print("H-%s is streaming [%s] H-%s " % (ip_pkt.src, tos, ip_pkt.dst))
                # if tos == 2**8 - 1:
                #
                if tos == 192: return
                if tos == 0 or tos not in self.tos_list:
                    # set flag == Flase
                    # if self.pre_tos != tos_flag:
                    #     self.same_tos_flag = False
                    self.tos_list.append(tos)
                    eth_type = pkt.get_protocols(ethernet.ethernet)[0].ethertype
                    self.shortest_forwarding(tos, msg, eth_type, ip_pkt.src, ip_pkt.dst)
                #
                # if tos == 2**8 - 1:
                #     del self.tos_list[:]