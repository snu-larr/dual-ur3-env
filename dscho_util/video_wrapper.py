# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import gym
from gym import Wrapper
from gym.wrappers.monitoring import video_recorder
from gym.wrappers.monitoring.video_recorder import VideoRecorder

class VideoWrapper(Wrapper):

  def __init__(self, env, base_path, base_name=None, new_video_every_reset=False, ur3_cam = True, custom_env = False):
    super(VideoWrapper, self).__init__(env)
    #dscho mod
    self.ur3_cam = ur3_cam
    self.custom_env = custom_env  

    self._base_path = base_path
    self._base_name = base_name

    self._new_video_every_reset = new_video_every_reset
    if self._new_video_every_reset:
      self._counter = 0
      self._recorder = None
       
      if self.ur3_cam:
        self._recorder_2 = None
      
    else:
      if self._base_name is not None:
        self._vid_name = os.path.join(self._base_path, self._base_name)
      else:
        self._vid_name = self._base_path
      # self._recorder = video_recorder.VideoRecorder(self.env, path=self._vid_name + '.mp4')
      self._recorder = MyVideoRecorderWrapper(self.env, path=self._vid_name + '.mp4')
       
      if self.ur3_cam:
        self._recorder_2 = MyVideoRecorderWrapper(self.env, path=self._vid_name +'_topview' '.mp4')

  def reset(self):
    if self._new_video_every_reset:
      if self._recorder is not None:
        self._recorder.close()
       
      if self.ur3_cam:
        if self._recorder_2 is not None:
          self._recorder_2.close()
      

      self._counter += 1
      if self._base_name is not None:
        self._vid_name = os.path.join(self._base_path, self._base_name + '_' + str(self._counter))
      else:
        self._vid_name = self._base_path + '_' + str(self._counter)

      # self._recorder = video_recorder.VideoRecorder(self.env, path=self._vid_name + '.mp4')
      self._recorder = MyVideoRecorderWrapper(self.env, path=self._vid_name + '.mp4')
       
      if self.ur3_cam:
        self._recorder_2 = MyVideoRecorderWrapper(self.env, path=self._vid_name +'_topview' '.mp4')

    return self.env.reset()

  def step(self, action):
     
    if self.ur3_cam:
      self._recorder.capture_frame(camera_name = 'frontview', custom_env=self.custom_env)
      self._recorder_2.capture_frame(camera_name = 'topview', custom_env=self.custom_env)
    else:
      self._recorder.capture_frame()
    return self.env.step(action)

  def close(self):
    self._recorder.encoder.proc.stdin.flush()
    self._recorder.close()
     
    if self.ur3_cam:
      self._recorder_2.encoder.proc.stdin.flush()
      self._recorder_2.close()

    return self.env.close()

  #dscho mod
  def __getattr__(self, name):
    return getattr(self.env, name)


from gym import error, logger
import numpy as np
class MyVideoRecorderWrapper(VideoRecorder):
  
  def __init__(self, env, path):
    super(MyVideoRecorderWrapper, self).__init__(env, path=path)
    self.frames_per_sec = int(np.round(1.0 / self.env.dt))
    self.output_frames_per_sec = self.frames_per_sec
    # self.frames_per_sec = env.metadata.get('video.frames_per_second', 30)
    # self.output_frames_per_sec = env.metadata.get('video.output_frames_per_second', self.frames_per_sec)
  
  # dscho mod, overrided
  def capture_frame(self, camera_id=None, camera_name=None, custom_env = False):
    """Render the given `env` and add the resulting frame to the video."""
    if not self.functional: return
    logger.debug('Capturing video frame: path=%s', self.path)

    render_mode = 'ansi' if self.ansi_mode else 'rgb_array'
    
    if custom_env:
      #dscho mod to include camera_id, camera_name
      frame = self.env.render(mode=render_mode, camera_id=camera_id, camera_name=camera_name)
    else:
      #dscho mod for fetch
      frame = self.env.render(mode=render_mode)

    if frame is None:
      if self._async:
        return
      else:
        # Indicates a bug in the environment: don't want to raise
        # an error here.
        logger.warn('Env returned None on render(). Disabling further rendering for video recorder by marking as disabled: path=%s metadata_path=%s', self.path, self.metadata_path)
        self.broken = True
    else:
      self.last_frame = frame
      if self.ansi_mode:
        self._encode_ansi_frame(frame)
      else:
        self._encode_image_frame(frame)
