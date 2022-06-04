#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :ryu
@File    :status_measure.py
@Author  :HLQ
@Date    :2022/2/18 上午10:38
'''

from __future__ import division

from operator import attrgetter
from ryu import cfg
from ryu.base import app_manager
from ryu.base.app_manager import lookup_service_brick
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import CONFIG_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
import configuration as CF
CONF = cfg.CONF
class MonitorDetection(app_manager.RyuApp):
    """
        MonitorDetection(Control plane)
          This class is mainly used for active distribution.
          It is used to query and save the perceived information of the link.
    """
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    def __init__(self, *args, **kwargs):
        super(MonitorDetection, self).__init__(*args, **kwargs)
        self.name = "monitor_module"
        self.topology_module = lookup_service_brick("topology_module")
        self.port_stats = {}       # save port status
        self.flow_stats = {}
        self.monitor_spawn = hub.spawn(self._monitor_detector)


    def _monitor_detector(self):
        """
            Cooperative process for detecting link state's change.
        """
        while True:
            hub.sleep(CF.MONITOR_PERIOD)
            if self.topology_module is None:
                self.topology_module = lookup_service_brick("topology_module")
            datapaths = dict(sorted(self.topology_module.datapaths.items()))      # sorted dpid
            for dp in datapaths.values():
                self._request_stats(dp)

    def _request_stats(self, datapath):
        """
            Active send out switch status request to query switch port state
        :param datapath:
        :return:
        """
        self.logger.debug('send stats request: %016x', datapath.id)
        ofproto = datapath.ofproto
        ofproto_parser = datapath.ofproto_parser
        req = ofproto_parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)


    def save_stats(self, dict_type, key, value, max_len=5):
        """
            Save link awareness information for calculating link state information.
        :param dict_type:
        :param key: tuple
        :param val: tuple
        :param max_len:
        :return:
        """
        if key not in dict_type:
            dict_type.setdefault(key, [])
        dict_type[key].append(value)
        if len(dict_type[key]) > max_len:
            dict_type[key].pop(0)


    def get_used_bandwidth(self, c_rx_bytes, p_rx_bytes, t):
        """
            Cal used_bw(another method[retain]).
        :param c_rx_bytes:
        :param p_rx_bytes:
        :param t:
        :return:
        """
        return (c_rx_bytes - p_rx_bytes) * 8 / 10 ** 6 / t

    def get_port_speed(self, cur_bytes, pre_bytes,  t):
        """
            Get port's speed.
        :param cur_bytes:
        :param pre_bytes:
        :param t:
        :return:
        """
        return (cur_bytes - pre_bytes ) * 8 / 10 ** 6 / t

    def get_sec_from_nsec(self, sec, nsec):
        """
            Transfer nsec to sec(10**9).
        :param sec:
        :param nsec:
        :return:
        """
        return sec + nsec / (10 ** 9)

    def get_interval_time(self, c_sec, c_nsec, p_sec, p_nsec):
        """
            Get period.
        :param c_sec:
        :param c_nsec:
        :param p_sec:
        :param p_nsec:
        :return:
        """
        return self.get_sec_from_nsec(c_sec, c_nsec) - self.get_sec_from_nsec(p_sec, p_nsec)


    def get_link_bandwidth_info_fun(self, key1, key2, link_bw=None):
        """
            Get the free_bw and used_bw of link.If link_bw is None to cal used_bw, otherwise cal free_bw.
        :param key1:
        :param key2:
        :param link_bw:
        :return:
        """
        # get time:  t1 #
        t1 = self.get_interval_time(self.port_stats[key1][-1][6],
                                   self.port_stats[key1][-1][7],
                                   self.port_stats[key1][-2][6],
                                   self.port_stats[key1][-2][7])
        # get port's speed: speed1 #
        speed1 = self.get_port_speed(
            self.port_stats[key1][-1][0] + self.port_stats[key1][-1][1],
            self.port_stats[key1][-2][0] + self.port_stats[key1][-2][1], t1)
        # get time:  t2 #
        t2 = self.get_interval_time(self.port_stats[key2][-1][6],
                                   self.port_stats[key2][-1][7],
                                   self.port_stats[key2][-2][6],
                                   self.port_stats[key2][-2][7])
        # get port's speed: speed2 #
        speed2 = self.get_port_speed(
            self.port_stats[key2][-1][0] + self.port_stats[key2][-1][1],
            self.port_stats[key2][-2][0] + self.port_stats[key2][-2][1], t2)
        # cal free_bw #
        if link_bw:
            return max(link_bw - max(speed1, speed2), 0)
        # cal used_bw #
        return max(speed1, speed2)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        """
            Detect flow status of link.
        """
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        self.flow_stats.setdefault(dpid, {})
        for stat in sorted([flow for flow in body if flow.priority == 1],
                           key=lambda flow: (flow.match.get('in_port'),
                                             flow.match.get('ipv4_dst'))):
            key = (stat.match['in_port'], stat.match.get('ipv4_dst'),
                   stat.instructions[0].actions[0].port)
            value = (stat.packet_count, stat.byte_count,
                     stat.duration_sec, stat.duration_nsec)
            self.save_stats(self.flow_stats[dpid], key, value, 5)

            if len(self.flow_stats[dpid][key]) > 1:
                t = self.get_interval_time(self.flow_stats[dpid][key][-1][2], self.flow_stats[dpid][key][-1][3],
                                           self.flow_stats[dpid][key][-2][2], self.flow_stats[dpid][key][-2][3])
                speed = self.get_port_speed(self.flow_stats[dpid][key][-1][1], self.flow_stats[dpid][key][-2][1], t)
                print("<flow>", value[1], self.flow_stats[dpid][key][-2][1], self.flow_stats[dpid][key][-1][1], speed)
            else:
                return

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        """
            Detect link port infos
        :param ev:
        :return:
        """
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        for stat in sorted(body, key=attrgetter('port_no')):
            port_no = stat.port_no
            if port_no != ofproto_v1_3.OFPP_LOCAL:              # 4294967294 -> hex 0xfffffffe
                key = (dpid, port_no)
                # port information #
                value = (stat.tx_bytes, stat.rx_bytes, stat.tx_packets, stat.rx_packets,
                         stat.rx_errors, stat.tx_errors, stat.duration_sec, stat.duration_nsec,
                         stat.tx_dropped, stat.rx_dropped, stat.rx_over_err, stat.rx_crc_err,
                         stat.rx_frame_err, stat.collisions)
                self.save_stats(self.port_stats, key, value)    # save link aware information
