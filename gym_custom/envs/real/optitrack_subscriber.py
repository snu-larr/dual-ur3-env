
import rospy
from geometry_msgs.msg import PoseStamped # PointStamped, TwistStamped,
# from std_msgs.msg import String, Float32MultiArray
# from std_srvs.srv import Empty, EmptyResponse, Trigger, TriggerResponse
import copy
import numpy as np
import time
class OptitrackSubscriber(object):
    def __init__(self, rate, id_list):
        # In ROS, nodes are uniquely named. If two nodes with the same
        # name are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
        rospy.init_node('listener', anonymous=True)
        self.id_list = id_list # ['ur3_table', 'block_6cm_1']
        self.data = {}
        for id in self.id_list:
            self.data[id] = {}
        # self.most_recent_data = {}
        
        self.optitrack_subscribers = [rospy.Subscriber('/optitrack/'+ id +'/poseStamped',
                                                    PoseStamped,
                                                    # PointStamped,
                                                    callback=self._callback,
                                                    callback_args= id,
                                                    ) for id  in self.id_list]
        self.rate = rospy.Rate(rate)
        # spin() simply keeps python from exiting until this node is stopped
        # rospy.spin()
    


    def _callback(self, data, id):
        # header: 
        #     seq: 307939
        #     stamp: 
        #         secs: 1627391013
        #         nsecs: 841725488
        #     frame_id: "/world"
        # global x,y,z, sequence, quat_x, quat_y, quat_z, quat_w

        sequence = data.header.seq # for checking whether get_optitrack_data method receive latest data
        x = data.pose.position.x
        y = data.pose.position.y
        z = data.pose.position.z
        quat_x = data.pose.orientation.x
        quat_y = data.pose.orientation.y
        quat_z = data.pose.orientation.z
        quat_w = data.pose.orientation.w
        
        self.data[id].update({'x' : x, 'y' : y, 'z' : z, 'quat_x' : quat_x, 'quat_y' : quat_y, 'quat_z' : quat_z, 'quat_w' : quat_w, 'sequence' : sequence})

    def get_optitrack_data(self): # why past data is not preserved?
        self.rate.sleep()
        data = copy.deepcopy(self.data)
        # if data is None:
        #     print('data is None!')
        #     data = copy.deepcopy(self.most_recent_data)
        # else:
        #     self.most_recent_data = copy.deepcopy(data)
        return data



def optitrack_subscriber_class_test():
    optitrack = OptitrackSubscriber(rate=30, id_list = ['ur3_table', 'block_6cm_1'])
    time.sleep(1)
    for i in range(100):
        before = time.time()
        data = optitrack.get_optitrack_data()
        ur3_table_data = data['ur3_table']
        block_6cm_1_data = data['block_6cm_1']
        # print(data)
        dt = time.time()-before
        Hz = 1/dt
        # print(' dt : {} t : {} :pos : {:10f} {:10f} {:10f} quat : {:10f} {:10f} {:10f} {:10f}'.format(dt, data['sequence'], data['x'],data['y'], data['z'], data['quat_x'], data['quat_y'], data['quat_z'], data['quat_w']))
        print('ur3_table dt : {} t : {} :pos : {:10f} {:10f} {:10f} quat : {:10f} {:10f} {:10f} {:10f}'.format(dt, ur3_table_data['sequence'], ur3_table_data['x'],ur3_table_data['y'], ur3_table_data['z'], ur3_table_data['quat_x'], ur3_table_data['quat_y'], ur3_table_data['quat_z'], ur3_table_data['quat_w']))
        print('block_6cm_1 dt : {} t : {} :pos : {:10f} {:10f} {:10f} quat : {:10f} {:10f} {:10f} {:10f}'.format(dt, block_6cm_1_data['sequence'], block_6cm_1_data['x'],block_6cm_1_data['y'], block_6cm_1_data['z'], block_6cm_1_data['quat_x'], block_6cm_1_data['quat_y'], block_6cm_1_data['quat_z'], block_6cm_1_data['quat_w']))
        
    

def optitrack_test():
    # Request  & Receive the data when you call the wait_for_message
    # NOTE : Do not recommend to use it. Peak occurs often.
    rospy.init_node('listener', anonymous=True)
    id = 'test_broom'
    data = rospy.wait_for_message('/optitrack/'+ id+'/poseStamped', PoseStamped, timeout=None)    
    print(data)

if __name__ == '__main__':
    # optitrack_test()
    optitrack_subscriber_class_test()