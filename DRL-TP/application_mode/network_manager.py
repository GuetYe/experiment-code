#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :ryu 
@File    :management_plane.py
@Author  :HLQ
@Date    :2022/2/18 上午10:39 
'''

from ryu.base import app_manager
from ryu.ofproto import ofproto_v1_3
from ryu.base.app_manager import lookup_service_brick
import configuration as CF



class ManagementPlane(app_manager.RyuApp):
    """
        ManagementPlane(Manage plane)
            This class is mainly used to calculate the state information of the link as the input of the knowledge plane. 
            The performance of the plane will process a large amount of data, so the performance of the plane determines 
            the performance of the intelligent routing algorithm.
    """
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(ManagementPlane, self).__init__(*args, **kwargs)
        self.name = "management_module"
        self.topology_module = lookup_service_brick("topology_module")
        self.monitor_module = lookup_service_brick("monitor_module")
        self.link_drop = {}        
        self.link_error = {}
        self.link_loss = {}
        self.link_bw = {}
        self.link_delay = {}
        self.link_used_bw = {}

    def get_traffic_matrix(self, paths):
        """
            Get network traffic matrix.(Used in Train Model(TM)).
        :param paths:
        :return:
        """
        path_metrics = {}
        if self.topology_module is None:
            self.topology_module = lookup_service_brick("topology_module")
        if self.monitor_module is None:
            self.monitor_module = lookup_service_brick("monitor_module")
        # obtain path dorps, errors, loss, bw, delay, used_bw #
        path_metrics.update(self.get_path_drop(paths))
        path_metrics.update(self.get_path_error(paths))
        path_metrics.update(self.get_path_loss(paths))
        path_metrics.update(self.get_path_bw(paths))
        path_metrics.update(self.get_path_delay(paths))
        path_metrics.update(self.get_path_used_bw(paths))
        return path_metrics


    def get_network_performance(self, paths):
        """
            Get network performance indicators.(Used in Test Model)
        :param paths:
        :return:
        """
        self.topology_module = lookup_service_brick("topology_module")
        self.monitor_module = lookup_service_brick("monitor_module")
        # get path delay, loss throughput #
        delay = self.get_path_delay(paths)["delay"]
        loss = self.get_path_loss(paths)["loss"]
        throughput = self.get_path_throughput(paths)["throughput"]
        return delay, loss, throughput


    def get_path_drop(self, paths):
        """
            Get path drops.
        :param paths: 
        :return: 
        """
        drop_list = []
        for path in paths:
            dropped = 0
            for i in range(len(path) - 1):
                try:
                    src_sw, dst_sw = int(path[i]), int(path[i + 1])
                    src_port, dst_port = self.topology_module.link_to_port[(src_sw, dst_sw)]
                    src_rx_dropped = self.monitor_module.port_stats[(src_sw, src_port)][-1][9]
                    src_tx_dropped = self.monitor_module.port_stats[(dst_sw, dst_port)][-1][8]
                    dst_rx_dropped = self.monitor_module.port_stats[(dst_sw, dst_port)][-1][9]
                    dst_tx_dropped = self.monitor_module.port_stats[(src_sw, src_port)][-1][8]
                    dropped += max(src_rx_dropped + src_tx_dropped, dst_rx_dropped + dst_tx_dropped)
                except:
                    continue
            drop_list.append(dropped)
        return {"drop": drop_list}

    def get_path_error(self, paths):
        """
            Get path errors.
        :param paths: 
        :return: 
        """
        error_list = []
        for path in paths:
            errors = 0
            for i in range(len(path) - 1):
                try:
                    src_sw, dst_sw = int(path[i]), int(path[i + 1])
                    src_port, dst_port = self.topology_module.link_to_port[(src_sw, dst_sw)]
                    src_rx_errors = self.monitor_module.port_stats[(src_sw, src_port)][-1][4]
                    src_tx_errors = self.monitor_module.port_stats[(dst_sw, dst_port)][-1][5]
                    dst_rx_errors = self.monitor_module.port_stats[(dst_sw, dst_port)][-1][4]
                    dst_tx_errors = self.monitor_module.port_stats[(src_sw, src_port)][-1][5]
                    errors += max(src_rx_errors + src_tx_errors, dst_rx_errors + dst_tx_errors)
                except:
                    continue
            error_list.append(errors)
        return {"error": error_list}


    def get_path_loss(self, paths):
        """
            Get path loss.
        :param paths: 
        :return: 
        """
        loss_list = []
        for path in paths:
            loss = 0.0
            for i in range(len(path) - 1):
                try:
                    src_sw, dst_sw = int(path[i]), int(path[i + 1])
                    src_port, dst_port = self.topology_module.link_to_port[(src_sw, dst_sw)]
                    src_rx_bytes = self.monitor_module.port_stats[(src_sw, src_port)][-1][1]
                    src_tx_bytes = self.monitor_module.port_stats[(dst_sw, dst_port)][-1][0]
                    dst_rx_bytes = self.monitor_module.port_stats[(dst_sw, dst_port)][-1][1]
                    dst_tx_bytes = self.monitor_module.port_stats[(src_sw, src_port)][-1][0]
                    loss = max((src_tx_bytes - src_rx_bytes) / float(src_tx_bytes),
                               (dst_tx_bytes - dst_rx_bytes) / float(dst_tx_bytes))

                except:
                    continue
            loss_list.append(loss)
        return {"loss": loss_list}



    def get_path_bw(self, paths):
        """
            Get path free_bw.
        :param paths: 
        :return: 
        """
        bw_list = []
        for path in paths:
            link_bw = CF.MAX_CAPACITY / 10 ** 3
            for i in range(len(path) - 1):
                try:
                    src_sw, dst_sw = int(path[i]), int(path[i + 1])
                    src_port, dst_port = self.topology_module.link_to_port[(src_sw, dst_sw)]
                    key1 = (src_sw, src_port)
                    key2 = (dst_sw, dst_port)
                    if (src_sw, dst_sw) in CF.LINK_INFOS:
                        bw = CF.LINK_INFOS[(src_sw, dst_sw)] / 10 ** 3
                    else:
                        bw = CF.MAX_CAPACITY / 10 ** 3
                    bw = self.monitor_module.get_link_bandwidth_info_fun(key1, key2, bw)
                    link_bw = min(link_bw, bw)
                except:
                    # print("lnk bw is Error : %s %s" % (src_sw, dst_sw))
                    continue
            bw_list.append(link_bw)
        return {"free_bw": bw_list}

    def get_path_delay(self, paths):
        """
            Get path delay.
        :param paths: 
        :return: 
        """
        delay_list = []
        for path in paths:
            link_delay = 0.0
            for i in range(len(path) - 1):
                try:
                    src_sw, dst_sw = int(path[i]), int(path[i + 1])
                    src_dst_delay = self.topology_module.graph[src_sw][dst_sw]["delay"]
                    link_delay += src_dst_delay
                except:
                    print("lnk delay is Error : %s %s" % (src_sw, dst_sw))
                    continue

            delay_list.append(link_delay)
        return {"delay": delay_list}


    def get_path_used_bw(self, paths):
        """
            Get path used_bw.
        :param paths: 
        :return: 
        """
        used_bw_list = []
        for path in paths:
            link_used_bw = 0
            for i in range(len(path) - 1):
                try:
                    src_sw, dst_sw = int(path[i]), int(path[i + 1])
                    src_port, dst_port = self.topology_module.link_to_port[(src_sw, dst_sw)]
                    key1 = (src_sw, src_port)
                    key2 = (dst_sw, dst_port)
                    used_bw = self.monitor_module.get_link_bandwidth_info_fun(key1, key2)
                    link_used_bw = max(link_used_bw, used_bw)
                except:
                    continue
            # print("<used_bw>", link_used_bw)
            used_bw_list.append(link_used_bw)
        return {"used_bw": used_bw_list}

    def get_path_throughput(self, paths):
        """
            Get path throughput.
        :param paths:
        :return:
        """
        throughput_list = []
        for path in paths:
            link_bw = CF.MAX_CAPACITY / 10 ** 3
            throughput = 0
            for i in range(len(path) - 1):
                try:
                    src_sw, dst_sw = int(path[i]), int(path[i + 1])
                    src_port, dst_port = self.topology_module.link_to_port[(src_sw, dst_sw)]
                    key1 = (src_sw, src_port)
                    key2 = (dst_sw, dst_port)
                    if (src_sw, dst_sw) in CF.LINK_INFOS:
                        bw = CF.LINK_INFOS[(src_sw, dst_sw)] / 10 ** 3
                    else:
                        bw = CF.MAX_CAPACITY / 10 ** 3
                    bw = self.monitor_module.get_link_bandwidth_info_fun(key1, key2, bw)
                    src_tx_bytes = self.monitor_module.port_stats[(dst_sw, dst_port)][-1][0]
                    link_bw = min(link_bw, bw)
                    throughput += src_tx_bytes / (link_bw * CF.MONITOR_PERIOD)
                except:
                    continue
            throughput_list.append(throughput)
        return {"throughput": throughput_list}
