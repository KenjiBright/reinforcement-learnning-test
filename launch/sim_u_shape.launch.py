import os
import re
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
import xacro

def _clean_urdf(xacro_file: str) -> str:
    raw = xacro.process_file(xacro_file).toxml()
    if raw.startswith('<?xml'):
        raw = raw[raw.index('?>') + 2:].lstrip()
    raw = re.sub(r'<!--.*?-->', '', raw, flags=re.DOTALL).strip()
    return raw

def generate_launch_description():
    os.environ.setdefault('ROS_LOCALHOST_ONLY', '1')
    os.environ.setdefault('ROS_DOMAIN_ID', '42')

    base_path  = os.path.expanduser('~/Desktop/my_4ws_robot')
    xacro_file = os.path.join(base_path, 'urdf', 'robot_4wd.urdf.xacro')
    world_file = os.path.join(base_path, 'worlds', 'maze_u_shape.world')

    robot_description = {'robot_description': _clean_urdf(xacro_file)}

    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose',
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so',
             '-s', 'libgazebo_ros_state.so',
             world_file],
        output='screen'
    )

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'robot_4wd',
                   '-x', '0.5', '-y', '0.0', '-z', '0.05'],
        output='screen'
    )

    load_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster']
    )
    load_wheel_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['wheel_controller']
    )

    return LaunchDescription([
        gazebo,
        node_robot_state_publisher,
        spawn_entity,
        RegisterEventHandler(event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[load_joint_state_broadcaster]
        )),
        RegisterEventHandler(event_handler=OnProcessExit(
            target_action=load_joint_state_broadcaster,
            on_exit=[load_wheel_controller]
        )),
    ])