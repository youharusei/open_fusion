"""
ROS node of goal location query client.
input semantic query and request the locaiton of top-1 instance,
send goal to move_base with actionlib client.

service: goal_location_query
request:
    string semantic query
response:
    bool success
    float location x
    float location y
    float size range_x
    float size range_y
"""
import rospy
from goal_location_query.srv import GoalLocationQuery, GoalLocationQueryRequest, GoalLocationQueryResponse
from actionlib import SimpleActionClient
from move_base_msgs.msg import MoveBaseActionGoal
from geometry_msgs.msg import PoseStamped


def main():
    rospy.init_node("goal_location_client_node")
    # initialize clients
    query_client = rospy.ServiceProxy("goal_location_client", GoalLocationQuery)
    move_base_client = SimpleActionClient("move_base", MoveBaseActionGoal)
    query_client.wait_for_service()
    move_base_client.wait_for_server()
    # query the location of top-1 semantic instance and send goal to move_base
    while not rospy.is_shutdown():
        query:str = input("[*] enter semantic query:")
        response:GoalLocationQueryResponse = query_client(GoalLocationQueryRequest(query))
        if response.success:
            goal = PoseStamped()
            goal.header.frame_id = "tsdf"
            goal.header.stamp = rospy.Time.now()
            goal.pose.position.x = response.x
            goal.pose.position.y = response.y
            move_base_client.cancel_all_goals()
            move_base_client.send_goal(goal)
            rospy.loginfo("[*] sent goal to move_base.")
        else:
            rospy.loginfo("[*] no suitable goal found for '%s'!", query)

if __name__ == "__main__":
    main()