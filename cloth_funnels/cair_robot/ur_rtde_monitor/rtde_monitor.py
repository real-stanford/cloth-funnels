from threading import Thread, Lock, Event
import copy
import logging
import time
import collections

import ur_rtde_monitor.rtde as rtde
import ur_rtde_monitor.rtde_variables as rtde_var


_log = logging.getLogger(rtde.LOGNAME)

class RTDEMonitor(Thread):
    def __init__(self, hostname, port=30004, maxlen=1250, frequency=125) -> None:
        super().__init__()
        
        self.connection = rtde.RTDE(hostname=hostname, port=port)
        self.deque = collections.deque(maxlen=maxlen)
        self.deque_lock = Lock()
        self.stop_event = Event()
        
        self.frequency = frequency
        self.variables = rtde_var.ROBOT_VARS

        self.start()

    def run(self):
        output_names = list(self.variables.keys())
        output_types = list(self.variables.values())

        con = self.connection
        con.connect()
        con.get_controller_version()
        if not con.send_output_setup(
            output_names, output_types, frequency=self.frequency):
            _log.error('Unable to configure output')
            return
        
        if not con.send_start():
            _log.error('Unable to start synchronization')
            return

        while not self.stop_event.is_set():
            state = None
            try:
                state = con.receive_buffered()
            except rtde.RTDEException as e:
                _log.error(e)
                con.disconnect()
                return
            if state is not None:
                with self.deque_lock:
                    self.deque.append(state.data)
            time.sleep(0)
        con.disconnect()
    
    def stop(self):
        self.stop_event.set()
        self.join()
    
    def __del__(self):
        self.stop()
    
    def __len__(self):
        return len(self.deque)

    def keys(self):
        return self.variables.keys()
    
    def get_latest_state(self) -> dict:
        result = None
        with self.deque_lock:
            result = copy.deepcopy(self.deque[-1])
        return result
    
    def get_latest_timestamp(self) -> float:
        state = self.get_latest_state()
        return state['timestamp']

    def get_states(self, start=float('-inf'), end=float('inf'), transpose=False) -> list:
        all_states = None
        with self.deque_lock:
            all_states = copy.deepcopy(list(self.deque))
        result_states = list(filter( 
            lambda x: start <= x['timestamp'] <= end,
            all_states))
        if transpose:
            result_states = self.transpose_states(result_states)
        return result_states

    def resize(self, size):
        with self.deque_lock:
            new_deque = collections.deque(self.deque, maxlen=size)
            self.deque = new_deque
    
    @staticmethod
    def transpose_states(states):
        import numpy as np
        assert(len(states) > 0)
        transposed_states = dict()
        for key in states[0].keys():
            transposed_states[key] = np.array([x[key] for x in states])
        return transposed_states
