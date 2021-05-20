import time

default_reid_loss_interval = 5  # 单位秒


class ReIDStates:
    def __init__(self, reid_loss_interval=default_reid_loss_interval):
        self.self_states = {}
        self.global_states = {'frame': 0, "interval": 1, "time": time.time()}
        self.reid_loss_interval = reid_loss_interval

    def next_frame(self):
        self.global_states["frame"] = (self.global_states["frame"] + 1) % 9999
        current_time = time.time()
        self.global_states["interval"] = current_time - self.global_states['time']
        self.global_states['time'] = current_time

    def __getitem__(self, reid):
        # 获取重识别状态
        if reid in self.self_states:
            self_state = self.self_states[reid]
            if (self.global_states['time'] - self_state['time']) > self.reid_loss_interval:
                self_state = {"time": time.time()}
                self.self_states[reid] = self_state
            else:
                self_state['time'] = time.time()
        else:
            self_state = {"time": time.time()}
            self.self_states[reid] = self_state
        return self_state

    def get(self, reid, field_name, default=0.0):
        self_state = self[reid]
        if field_name in self_state:
            value = self_state[field_name]
        else:
            value = default
            self_state[field_name] = value
        return value

    def get_global_interval(self):
        return self.global_states['interval']

    def smooth_set(self, reid, field_name, new_value, c=0.5, init_value=0.0):
        self_state = self[reid]
        old_value = self_state.get(field_name, init_value)
        self_state[field_name] = c * new_value + (1 - c) * old_value
        return self_state[field_name]

    def timer_set(self, reid, timer_name):
        self_state = self[reid]
        timer = self_state.get(timer_name, 0)
        timer += self.get_global_interval()
        self_state[timer_name] = timer
        return timer

    def timer_reset(self, reid, timer_name):
        self_state = self[reid]
        self_state[timer_name] = 0
