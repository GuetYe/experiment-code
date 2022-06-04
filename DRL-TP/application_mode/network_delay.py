#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :ryu
@File    :delay_measure.py
@Author  :HLQ
@Date    :2022/2/18 上午10:37
'''


from __future__ import division
import time
from ryu import cfg
from ryu.base import app_manager
from ryu.base.app_manager import lookup_service_brick
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.topology.switches import Switches
from ryu.topology.switches import LLDPPacket
from collections import  defaultdict
import configuration as CF

CONF = cfg.CONF


class DelayMeasure(app_manager.RyuApp):
    """
        DelayMeasure(Control Plane)
            This class is used to monitor network link delay.
    """

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(DelayMeasure, self).__init__(*args, **kwargs)
        self.name = "delay_module"
        self.sending_echo_request_interval = 0.05
        # Get the active object of swicthes and awareness module.
        # So that this module can use their data.
        self.sw_module = lookup_service_brick("switches")
        self.topology_module = lookup_service_brick("topology_module")
        self.echo_delay = {}
        self.lldp_delay = defaultdict(dict)
        self.measure_thread = hub.spawn(self._delay_detector)


    def _delay_detector(self):
        """
             Cooperative process for detecting link delay's change.
        :return:
        """
        while True:
            if self.topology_module is None:
                self.topology_module = lookup_service_brick("topology_module")
            self._send_echo_request()
            self.create_delay_graph()
            hub.sleep(CF.DELAY_DETECTING_PERIOD)


    def create_delay_graph(self):
        """
            Store the network link delay in the graph.
        :return:
        """
        try:
            for src, dst in self.topology_module.graph.edges():
                if src == dst:
                    self.topology_module.graph[src][dst]["delay"] = 0.0
                    continue
                link_delay = self.get_link_delay(src, dst)
                if link_delay:
                    self.topology_module.graph[src][dst]["delay"] = link_delay
        except:
            if self.topology_module is None:
                self.topology_module = lookup_service_brick("topology_module")
            return


    def get_link_delay(self, src_sw, dst_sw):
        """
            Get link delay.
                        Controller
                        |        |
        src echo latency|        |dst echo latency
                        |        |
                   SwitchA-------SwitchB

                    fwd_delay--->
                        <----reply_delay
            delay = (forward delay + reply delay - src datapath's echo latency
        """
        try:
            fwd_delay = self.lldp_delay[src_sw][dst_sw]
            re_delay = self.lldp_delay[dst_sw][src_sw]
            src_latency = self.echo_delay[src_sw]
            dst_latency = self.echo_delay[dst_sw]
            delay = (fwd_delay + re_delay - src_latency - dst_latency) / 2
            return max(delay, 0)
        except:
            return 0


    def _send_echo_request(self):
        """
            Send echo request to query the delay from the switch to the controller.
        :return:
        """
        try:
            for datapath in self.topology_module.datapaths.values():
                ofproto_parser = datapath.ofproto_parser
                echo_req = ofproto_parser.OFPEchoRequest(datapath, data="%.20f" % time.time())
                datapath.send_msg(echo_req)
                hub.sleep(self.sending_echo_request_interval)
        except:
            return

    @set_ev_cls(ofp_event.EventOFPEchoReply, MAIN_DISPATCHER)
    def echo_reply_handler(self, ev):
        """
            Handle the echo reply msg, and get the latency of link.
        :param ev:
        :return:
        """
        try:
            msg = ev.msg
            datapath = msg.datapath
            echo_delay = float("%.20f" % (time.time() - eval(msg.data)))
            self.echo_delay[datapath.id] = echo_delay
        except:
            return

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """
            Handler network's pkt to cal link delay.
        :param ev:
        :return:
        """
        msg = ev.msg
        try:
            src_dpid, src_port_no = LLDPPacket.lldp_parse(msg.data)
            dst_dpid = msg.datapath.id
            if self.sw_module is None:
                self.sw_module = lookup_service_brick("switches")
            for port in self.sw_module.ports.keys():                    # ryu.topology.switches.Port object
                if src_dpid == port.dpid and src_port_no == port.port_no:
                    delay = self.sw_module.ports[port].delay
                    self.lldp_delay[src_dpid][dst_dpid] = delay

        except LLDPPacket.LLDPUnknownFormat as e:
            return




