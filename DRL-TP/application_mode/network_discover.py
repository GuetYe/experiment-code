#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :ryu
@File    :topo_discover.py
@Author  :HLQ
@Date    :2022/2/18 上午10:35
'''

import time
# import json
import pickle
# import cPickle as pickle
import socket
import networkx as nx
from ryu import cfg
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import CONFIG_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import arp, ipv4
from ryu.lib import hub
from ryu.topology import event, switches
from ryu.topology.api import get_switch, get_link
from thread import start_new_thread, exit_thread
import configuration as CF
# from process import StoreProcess, TakeProcess

CONF = cfg.CONF


class TopoDiscover(app_manager.RyuApp):
    """
        TopoDiscover(Data Plane)
            This class is mainly used to discover network topology information and
            forward data according to the relevant forwarding instructions of the control plane

    """
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(TopoDiscover, self).__init__(*args, **kwargs)
        self.topology_api_app = self
        self.name = "topology_module"
        self.datapaths = {}
        # intra switch's info #
        self.graph = nx.DiGraph()
        self.link_to_port = {}
        self.dpid_to_port = {}
        self.mapping_ip_mac = {}
        self.all_switch_port = {}
        self.all_host_port = {}
        self.all_link_port = {}
        self.arp_pkt_detect = True  # flag used to stop detecting ARP packets(modified during flow)
        self.arp_pkt_send = False  # flag the ARP is sending(used to start detecting candidate paths)
        # self.discover_spawn = hub.spawn(self._discover_topology)


    def _discover_topology(self):
        """
           link discover spawn
        :return:
        """
        while True:
            self.show_static_infos()
            hub.sleep(20)

    def get_dijkstra_path(self, src, dst, weight="weight", k=1):
        """
            Great DIJKSTRA path.
        :return
        """
        try:
            dijkstra_path = nx.dijkstra_path(self.graph, source=src, target=dst, weight=weight)
            return dijkstra_path
        except:
            print ("\033[37;1m No path between %s and %s \033[0m" % (src, dst))

    def get_ospf_path(self, src, dst, weight="delay", k=1):
        """
            Get OSPF path.
        :param shortest_path:
        :param src:
        :param dst:
        :param weight:
        :param k:
        :return:
        """
        try:
            ospf_path = nx.dijkstra_path(self.graph, source=src, target=dst, weight=weight)
            return ospf_path
        except:
            print("\033[37;1m No path between %s and %s \033[0m" % (src, dst))



    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """
            Detect switch.
        :param ev:
        :return:
        """
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        msg = ev.msg
        self.logger.info("switch:%s connected", datapath.id)
        if datapath.id not in self.datapaths:
            self.datapaths[datapath.id] = datapath
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, dp, p, match, actions, idle_timeout=0, hard_timeout=0):
        """
            Switch establishes a flow table
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

    def get_dpid_port_from_mapping(self, host_ip):
        """

        :param host_ip:
        :return: dpid(sw)
        """
        for key in self.mapping_ip_mac.keys():
            if self.mapping_ip_mac[key][0] == host_ip:
                return key
        self.logger.info("%s location is not found." % host_ip)
        return None

    def get_sw_from_mapping(self, dpid, in_port, src, dst):
        """

        :param dpid:
        :param in_port:
        :param src:
        :param dst:
        :return: (src_sw, dst_sw)
        """
        src_sw = dpid
        dst_sw = None
        src_location = self.get_dpid_port_from_mapping(src)
        if in_port in self.all_host_port[dpid]:
            if (dpid, in_port) == src_location:
                src_sw = src_location[0]
            else:
                return None
        dst_location = self.get_dpid_port_from_mapping(dst)
        if dst_location:
            dst_sw = dst_location[0]

        return src_sw, dst_sw

    def get_port_pair_from_link(self, src_dpid, dst_dpid):
        """
        :param src_dpid:
        :param dst_dpid:
        :return: (src_port, dst_port)
        """
        if (src_dpid, dst_dpid) in self.link_to_port:
            return self.link_to_port[(src_dpid, dst_dpid)]
        else:
            self.logger.info("dpid:%s->dpid:%s is not in links" % (
                src_dpid, dst_dpid))
            return None

    def get_dst_port_from_mapping(self, dst_ip):
        """
        :param dst_ip:
        :return: port
        """
        for key in self.mapping_ip_mac.keys():
            if dst_ip == self.mapping_ip_mac[key][0]:
                dst_port = key[1]
                return dst_port
        return None

    def create_network_graph(self):
        """
            create the graph
        :return:
        """
        link_list = self.link_to_port.keys()
        for src in self.all_switch_port.keys():
            for dst in self.all_switch_port.keys():
                if src == dst:
                    self.graph.add_edge(src, dst, weight=0)
                elif (src, dst) in link_list:
                    self.graph.add_edge(src, dst, weight=1)
        return self.graph

    def create_all_switch_ports(self):
        """
            Set all port's info.
        :return:
        """
        switch_list = get_switch(self.topology_api_app, None)
        for sw in switch_list:
            dpid = sw.dp.id
            self.all_switch_port.setdefault(dpid, set())
            self.all_link_port.setdefault(dpid, set())
            self.all_host_port.setdefault(dpid, set())
            for p in sw.ports:
                self.all_switch_port[dpid].add(p.port_no)

    def create_all_link_ports(self):
        """
        :return:
        """
        link_list = get_link(self.topology_api_app, None)
        for link in link_list:
            src = link.src
            dst = link.dst
            self.link_to_port[
                (src.dpid, dst.dpid)] = (src.port_no, dst.port_no)
            self.dpid_to_port[(src.dpid, src.port_no)] = (dst.dpid, dst.port_no)
            if link.src.dpid in self.all_link_port.keys():
                self.all_link_port[link.src.dpid].add(link.src.port_no)
            if link.dst.dpid in self.all_link_port.keys():
                self.all_link_port[link.dst.dpid].add(link.dst.port_no)


    def create_all_host_ports(self):
        """
        :return:
        """
        for sw in self.all_switch_port:
            all_ports = self.all_switch_port[sw]
            link_ports = self.all_link_port[sw]
            host_ports = all_ports - link_ports
            if host_ports:
                self.all_host_port[sw] = host_ports

    events = [event.EventSwitchEnter,
              event.EventSwitchLeave, event.EventPortAdd,
              event.EventPortDelete, event.EventPortModify,
              event.EventLinkAdd, event.EventLinkDelete]
    @set_ev_cls(events)
    def get_topology(self, ev):
        """
            Get network's topo infos.
        :param ev:
        :return:
        """
        self.create_all_switch_ports()
        self.create_all_link_ports()
        self.create_all_host_ports()
        self.create_network_graph()

    def create_mapping_ip_mac(self, dpid, in_port, ip, mac):
        """
            Register access host info into access table.
        :param dpid:
        :param in_port:
        :param ip:
        :param mac:
        :return:
        """
        if in_port in self.all_host_port[dpid]:
            if (dpid, in_port) in self.mapping_ip_mac:
                if self.mapping_ip_mac[(dpid, in_port)] == (ip, mac):
                    return
                else:
                    self.mapping_ip_mac[(dpid, in_port)] = (ip, mac)
                    return
            else:
                self.mapping_ip_mac.setdefault((dpid, in_port), None)
                self.mapping_ip_mac[(dpid, in_port)] = (ip, mac)
                return

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """
            Handler the packet in packet, and register the access info.
        :param ev:
        :return:
        """
        msg = ev.msg
        datapath = msg.datapath
        in_port = msg.match['in_port']
        pkt = packet.Packet(msg.data)
        arp_pkt = pkt.get_protocol(arp.arp)
        if arp_pkt and self.arp_pkt_detect:
            self.arp_pkt_send = True
            arp_src_ip = arp_pkt.src_ip
            mac = arp_pkt.src_mac
            self.inter_info_falg = False
            self.create_mapping_ip_mac(datapath.id, in_port, arp_src_ip, mac)

    def show_static_infos(self):
        """"""
        # print(" ============ ====== ====== Host port ======= ====== ============ ")
        # print(self.all_host_port)

        # print(" ============ ====== ====== link to port ======= ====== ============ ")
        # print(self.link_to_port)

        # print(" ============ ====== ====== link list ======= ====== ============ ")
        # print(self.link_to_port)

        # print(" ============ ====== ====== mapping host ======= ====== ============ ")
        # print("the mapping len is %s " % len(self.mapping_ip_mac.keys()))

        # print(" ============ ====== ====== inter path ======= ====== ============ ")
        # print(self.optimal_inter_path)
        #
        # print(" ============ ====== ====== inter delay ======= ====== ============ ")
        # print(self.optimal_inter_path_delay)

        # print(" ============ ====== ====== inter link to port ======= ====== ============ ")
        # print(self.inter_link_to_port)

        # print(" ============ ====== ====== Inter port ======= ====== ============ ")
        # print(self.all_inter_dpid)
        print(" ============ ====== ====== Graph ======= ====== ============ ")
        for src, dst in self.graph.edges:
            print(src, dst, self.graph[src][dst])

