sensor_toggles = {}
function sysCall_init()
    -- ########### SETTING THE DEFAULT CONFIG
    sensor_toggles["line_left"] = true
    sensor_toggles["line_right"] = true
    sensor_toggles["line_center"] = true
    -- prox
    sensor_toggles["prox_front"] = true
    sensor_toggles["prox_back"] = true
    sensor_toggles["prox_front_left"] = true
    sensor_toggles["prox_back_left"] = true
    sensor_toggles["prox_front_right"] = true
    sensor_toggles["prox_back_right"] = true
    sensor_toggles["prox_front_left_45"] = true
    sensor_toggles["prox_back_left_45"] = true
    sensor_toggles["prox_font_right_45"] = true
    sensor_toggles["prox_font_right_45"] = true
    -- lidar
    sensor_toggles["lidar"] = true
    -- cam
    sensor_toggles["camera_rgb"] = true
    sensor_toggles["camera_depth"] = true
    -- render
    sensor_toggles["render_line"] = true
    sensor_toggles["render_prox"] = true
    sensor_toggles["render_lidar"] = true
    sensor_toggles["render_camera"] = true
    --########### GETTING COMPONENT HANDLES #######################
    --Main robot handle
    base_frame=sim.getObject("/Ack_robot")
    --Steering handles
    steeringLeft=sim.getObject('/Ack_robot/stear_front_right')
    steeringRight=sim.getObject('/Ack_robot/stear_front_left')
    
    
    --Motor handles
    motorLeft=sim.getObject('/Ack_robot/motor_front_right')
    motorRight=sim.getObject('/Ack_robot/motor_front_left')    
    
    
    --Sensor handle for each of the prox sensors
    proxFrontLeft = sim.getObject("./Proximity_front_left") 
    proxFront = sim.getObject("./Proximity_front") 
    proxFrontRight = sim.getObject("./Proximity_front_right") 
    proxBack = sim.getObject("./Proximity_back") 
    proxBackLeft = sim.getObject("./Proximity_back_left") 
    proxBackRight = sim.getObject("./Proximity_back_right") 
    proxBackRightRight = sim.getObject("./Proximity_back_RightRight") 
    proxBackLeftLeft = sim.getObject("./Proximity_back_LeftLeft") 
    proxFrontRightRight = sim.getObject("./Proximity_front_RightRight") 
    proxFrontLeftLeft = sim.getObject("./Proximity_front_leftLeft") 

    --Sensor handle for each of the line sensors
    line_left = sim.getObject("./line_right") 
    line_middle = sim.getObject("./line_middle") 
    line_right = sim.getObject("./line_left") 

    --Sensor handle for the RGB camara
    RGB_camara=sim.getObject('./RGB_camara')
    
    --Sensor handle for the Depth Camara
    depth_camara = sim.getObject('./depth_camara')
    Hokuyo_LIDAR = sim.getObject('./fastHokuyo_ROS')


    --Handle for the robot
    robotHandle=sim.getObject('.')
    --########### CREATING THE ROS INTERFACE #######################
    
    if simROS then
         sim.addLog(sim.verbosity_scriptinfos,"ROS interface was found.")
         --###############################################
         --Create a set of subscribers to get data from ROS
         motor_sub=simROS.subscribe('/cmd_vel_ack', 'demo_programs/ackermann_control', 'Motor_control_callback')
         simROS.subscriberTreatUInt8ArrayAsString(motor_sub)
         
        --###############################################
        --Create a set of publishers to set data to ROS
        --Creating for prox sensors
        prox_pub=simROS.advertise('/cop/prox_sensors', 'demo_programs/prox_sensor')
        simROS.publisherTreatUInt8ArrayAsString(prox_pub) 
        
        --Creating for line sensors
        line_pub=simROS.advertise('/cop/line_sensors', 'demo_programs/line_sensor')
        simROS.publisherTreatUInt8ArrayAsString(line_pub)
        
        --Creating publisher for RGB camara
        RGB_pub=simROS.advertise('/cop/rgb_image', 'sensor_msgs/Image')
        simROS.publisherTreatUInt8ArrayAsString(RGB_pub)
        
        --Creating publisher for the Depth camara
        depth_pub =simROS.advertise('/cop/depth_image', 'sensor_msgs/Image')
        simROS.publisherTreatUInt8ArrayAsString(depth_pub) 
                        -- Creating publisher for the both the depth and rbg camara info (both the same cam)
                        cam_into_pub = simROS.advertise('/cop/camera_info',
                        'sensor_msgs/CameraInfo')
        simROS.publisherTreatUInt8ArrayAsString(cam_into_pub)

        --Creating publisher for the pose topic
        pose_pub = simROS.advertise('/cop/pose','geometry_msgs/Pose')
        simROS.publisherTreatUInt8ArrayAsString(pose_pub) 
        
        --Creating publisher for the twist topic
        twist_pub = simROS.advertise('/cop/twist','geometry_msgs/Twist')

    
    else
        sim.addLog(sim.verbosity_scripterrors,"ROS interface was not found. Cannot run.")  
    end
    
    
end
-- #################################### SENSOR CONTROL
-- Checks the data string for each sensor 
function get_sensor_config()
    local sensor_signals = {}
    -- Getting each of the sensor configs
    sensor_signals["line_left"] = sim.getStringSignal("line_left")
    sensor_signals["line_right"] = sim.getStringSignal("line_right")
    sensor_signals["line_center"] = sim.getStringSignal("line_center")
    -- Prox
    sensor_signals["prox_front"] = sim.getStringSignal("prox_front")
    sensor_signals["prox_back"] = sim.getStringSignal("prox_back")
    sensor_signals["prox_front_left"] = sim.getStringSignal("prox_front_left")
    sensor_signals["prox_back_left"] = sim.getStringSignal("prox_back_left")
    sensor_signals["prox_front_right"] = sim.getStringSignal("prox_front_right")
    sensor_signals["prox_back_right"] = sim.getStringSignal("prox_back_right")
    sensor_signals["prox_front_left_45"] =
        sim.getStringSignal("prox_front_left_45")
    sensor_signals["prox_back_left_45"] =
        sim.getStringSignal("prox_back_left_45")
    sensor_signals["prox_font_right_45"] =
        sim.getStringSignal("prox_font_right_45")
    sensor_signals["prox_back_right_45"] =
        sim.getStringSignal("prox_back_right_45")
    -- Lidar 

    -- Camera
    sensor_signals["camera_rgb"] = sim.getStringSignal("camera_rgb")
    sensor_signals["camera_depth"] = sim.getStringSignal("camera_depth")

    sensor_signals["render_line"] = sim.getStringSignal("render_line")
    sensor_signals["render_prox"] = sim.getStringSignal("render_prox")
    sensor_signals["render_camera"] = sim.getStringSignal("render_camera")
    
    sensor_signals["prox_front"] = sim.getStringSignal("prox_front")
    for sensor_signal, data in pairs(sensor_signals) do
        if (data) then
            unpack_data = sim.unpackTable(data)
            if unpack_data[1] == 2 then
                -- print(sensor_signal.." : True")
                sensor_toggles[sensor_signal] = true
--
            elseif unpack_data[1] == 0 then
                sensor_toggles[sensor_signal] = false
                --print(sensor_signal.." : False")
            end
        end
    end
    --print(sensor_toggles)
    return sensor_toggles
end

--Sening calls
function sysCall_sensing()
    --Sends the TF (transformation) of the robot
    -- Send the robot's transform:
    simROS.sendTransform(getTransformStamped(robotHandle,'Ack_robot',-1,'world'))
    sensor_toggles = get_sensor_config()
    timestamp = simROS.getTime()
    transformations = {}
    transformations[1] = (getTransformStamped(robotHandle, 'Ack_robot',
    -1, 'world'))



    steeringLeft=sim.getObject('/Ack_robot/stear_front_right')
    steeringRight=sim.getObject('/Ack_robot/stear_front_left')
    
    
    --Motor handles
    motorLeft=sim.getObject('/Ack_robot/motor_front_right')
    motorRight=sim.getObject('/Ack_robot/motor_front_left')    



-- Defining TF for motors
transformations[2] = (getTransformStamped(  steeringLeft,  '/Ack_robot/stear_front_right',
    robotHandle, 'Ack_robot'))
transformations[3] = (getTransformStamped( steeringRight, './Ack_robot/stear_front_left',
    robotHandle, 'Ack_robot'))

transformations[4] = (getTransformStamped(motorLeft, '/Ack_robot/stear_front_right',
    robotHandle, 'Ack_robot'))
transformations[5] = (getTransformStamped(motorLeft, './Ack_robot/stear_front_left',
    robotHandle, 'Ack_robot'))
--Defining TF for Line sensors 
transformations[6] = (getTransformStamped(line_left, 'line_right',
    robotHandle, 'Ack_robot'))
transformations[7] = (getTransformStamped(line_middle, 'line_middle',
    robotHandle, 'Ack_robot'))
transformations[8] = (getTransformStamped(line_right, 'line_left',
    robotHandle, 'Ack_robot'))

-- defining TF for camara
transformations[9] = (getTransformStamped(RGB_camara, 'RGB_camara',
    robotHandle, 'Ack_robot'))
transformations[10] = (getTransformStamped(depth_camara, 'depth_camara',
    robotHandle, 'Ack_robot'))

-- defining TF for lidar
transformations[11] = (getTransformStamped(Hokuyo_LIDAR, 'fastHokuyo_ROS',
    robotHandle, 'Ack_robot'))
-- Defining TF for prox sensors
transformations[12] = (getTransformStamped(proxFrontLeft,
     'Proximity_front_left',
     robotHandle, 'Ack_robot'))
transformations[13] = (getTransformStamped(proxFront, 'Proximity_front',
     robotHandle, 'Ack_robot'))
transformations[14] = (getTransformStamped(proxFrontRight,
     'Proximity_front_right',
     robotHandle, 'Ack_robot'))
transformations[15] = (getTransformStamped(proxBack, 'Proximity_back',
     robotHandle, 'Ack_robot'))
transformations[16] = (getTransformStamped(proxBackLeft,
     'Proximity_back_left',
     robotHandle, 'Ack_robot'))
transformations[17] = (getTransformStamped(proxBackRight,
     'Proximity_back_right',
     robotHandle, 'Ack_robot'))

    publish_range_sensor()
    publish_line_sensors()
    publish_RBG_camera(timestamp)
    publish_depth_camara(timestamp)
    publish_pose_data()
    publish_twist_data()
    publish_depth_camara_info(timestamp)
    simROS.sendTransforms(transformations)
end

-- Acuation calls

-- ####################################
-- PUBLISHING
-- twiest data publish
function publish_twist_data()
linv,angv = sim.getObjectVelocity(base_frame)
--print(linv)
--print(angv)
twist = {}
twist['linear'] = {x=linv[1],y=linv[2],z=linv[3]}
twist['angular'] =  {x=angv[1],y=angv[2],z=angv[3]}
simROS.publish(twist_pub,twist)

end

--pose data publishing
function publish_pose_data()
    p=sim.getObjectPosition(base_frame,-1) 
    o=sim.getObjectQuaternion(base_frame,-1)
    pose = {}
    pose['position']={x=p[1],y=p[2],z=p[3]}
    pose['orientation']={x=o[1],y=o[2],z=o[3],w=o[4]}
    --print(pose)
    simROS.publish(pose_pub,pose)
end
-- Depth camera publish
function publish_depth_camara_info(timestamp)
    local _, angle = sim.getObjectFloatParam(depth_camara,
                                             sim.visionfloatparam_perspective_angle)
    local res = sim.getVisionSensorResolution(depth_camara)
    local D = {}
    local K = {}
    local R = {}
    local P = {}
    local width = res[1]
    local height = res[2]
    local f_x = (width / 2.) / math.tan(height / 2.)
    local f_y = f_x
    
    -- Setting the matrix paramiters
    -- Setting the distortion paramisters  (as this is a sim there is none)
    D = {0, 0, 0, 0, 0}
    -- Setting the camara matrix
    K = {f_x, 0, width / 2, 0, f_y, height / 2, 0, 0, 1}
    -- Setting the rectification matrix
    R = {1, 0, 0, 0, 1, 0, 0, 0, 1}
    -- Setting the projection matrix
    -- P = {K[0], 1, K[2], 0, 0, K[4], K[5], 0, 0, 0, 1, 0}
    P = {f_x, 1, width / 2, 0, 0, f_y, height / 2, 0, 0, 0, 1, 0}
    depth_info = {}
    -- Gets the header data
    depth_info['header'] = {stamp = timestamp, frame_id = "Ack_robot"}
    -- gets the width ahd the hight
    depth_info['height'] = height
    depth_info['width'] = width
    -- Geets the distorion model name
    depth_info['distortion_model'] = "plumb_bob"
    depth_info['D'] = D
    depth_info['K'] = K
    depth_info['R'] = R
    depth_info['P'] = P
    -- print( depth_info['P'])
    simROS.publish(cam_into_pub, depth_info)
    end
    -- Depth camera publish
    function publish_depth_camara(timestamp)
        if (sensor_toggles["camera_depth"]) then
        local res, nearClippingPlane = sim.getObjectFloatParameter(depth_camara,
                                                                   sim.visionfloatparam_near_clipping)
        local res, farClippingPlane = sim.getObjectFloatParameter(depth_camara,
                                                                  sim.visionfloatparam_far_clipping)
                                                                  local data = sim.getVisionSensorDepthBuffer(depth_camara +
                                                                  sim.handleflag_codedstring)
                  local res, nearClippingPlane = sim.getObjectFloatParameter(depth_camara,
                                                                             sim.visionfloatparam_near_clipping)
                  local res, farClippingPlane = sim.getObjectFloatParameter(depth_camara,
                                                                            sim.visionfloatparam_far_clipping)
                  nearClippingPlane = nearClippingPlane * 100
                  farClippingPlane = farClippingPlane * 100
                  data = sim.transformBuffer(data, sim.buffer_float,
                                             farClippingPlane - nearClippingPlane,
                                             nearClippingPlane, sim.buffer_uint16)
                  local res = sim.getVisionSensorResolution(depth_camara)
                  width, height = res[1], res[2]
                  depth_d = {}
                  depth_d['header'] = {stamp = timestamp, frame_id = "Ack_robot"}
                  depth_d['height'] = height
                  depth_d['width'] = width
                  depth_d['encoding'] = '16UC1'
                  depth_d['is_bigendian'] = 1
                  depth_d['step'] = width * 2
                  depth_d['data'] = data
                  simROS.publish(depth_pub, depth_d)
              
        end
    
    end


-- RGB camera publish
function publish_RBG_camera(timestamp)
    if (sensor_toggles["render_camera"]) then
        sim.setObjectInt32Param(RGB_camara,sim.objintparam_visibility_layer,1)
        sim.setObjectInt32Param(depth_camara,sim.objintparam_visibility_layer,1)
    else
        sim.setObjectInt32Param(RGB_camara,sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Param(depth_camara,sim.objintparam_visibility_layer,0)
    end 


    if (sensor_toggles["camera_rgb"]) then
        local data, w, h = sim.getVisionSensorCharImage(RGB_camara)
        d = {}
        d['header'] = {stamp = timestamp, frame_id = "Ack_robot"}
        d['height'] = h
        d['width'] = w
        d['encoding'] = 'rgb8'
        d['is_bigendian'] = 1
        d['step'] = w * 3
        d['data'] = data
        simROS.publish(RGB_pub, d)
    end
end

-- Line sensor publish
function publish_line_sensors()
    sensor_map = {}
    sensor_map['line_left'] = false
    sensor_map['line_right'] = false
    sensor_map['line_middle'] = false

    if (sensor_toggles["line_left"]) then
        sensor_map['line_left'] = int_to_bool((sim.readVisionSensor(line_right)))
    end
    if sensor_toggles["line_right"] then
        sensor_map['line_right'] = int_to_bool((sim.readVisionSensor(line_left)))
    end
    if sensor_toggles["line_center"] then
        sensor_map['line_middle'] = int_to_bool((sim.readVisionSensor(line_middle)))
    end


    if (sensor_toggles["render_line"]) then
        sim.setObjectInt32Param(line_right,sim.objintparam_visibility_layer,1)
        sim.setObjectInt32Param(line_left,sim.objintparam_visibility_layer,1)
        sim.setObjectInt32Param(line_middle,sim.objintparam_visibility_layer,1)
    else
        sim.setObjectInt32Param(line_right,sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Param(line_left,sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Param(line_middle,sim.objintparam_visibility_layer,0)
    end 
    
    -- print(sensor_map)
    simROS.publish(line_pub, sensor_map)
end
-- Publish the range sensor to ROS
function publish_range_sensor()
    -- TODO : Clean up this function
    local proxfr, proxfl, proxf, proxbl, proxbr, proxb

    proxfr = -1
    proxfl = -1
    proxf = -1
    proxbr = -1
    proxbl = -1
    proxb = -1
    proxbrr = -1
    proxbll = -1
    proxfrr = -1
    proxfll = -1

    if (sensor_toggles["prox_front"]) then
        _, proxf = sim.readProximitySensor(proxFront)
    end
    if sensor_toggles["prox_back"] then
        _, proxb = sim.readProximitySensor(proxBack)
    end
    if sensor_toggles["prox_front_left"] then
        _, proxfl = sim.readProximitySensor(proxFrontLeft)
    end
    if sensor_toggles["prox_back_left"] then
        _, proxbl = sim.readProximitySensor(proxBackLeft)
    end
    if sensor_toggles["prox_front_right"] then
        _, proxfr = sim.readProximitySensor(proxFrontRight)
    end
    if sensor_toggles["prox_back_right"] then
        _, proxbr = sim.readProximitySensor(proxBackRight)
    end
    if sensor_toggles["prox_front_left_45"] then
        _, proxfll = sim.readProximitySensor(proxFrontLeftLeft)
    end
    if sensor_toggles["prox_back_left_45"] then
        _, proxbll = sim.readProximitySensor(proxBackLeftLeft)
    end
    if sensor_toggles["prox_back_right_45"] then
        _, proxbrr = sim.readProximitySensor(proxBackRightRight)
    end
    if sensor_toggles["prox_font_right_45"] then
        _, proxfrr = sim.readProximitySensor(proxFrontRightRight)
    end


    --Controlling if the render renders the sensors
    if (sensor_toggles["render_prox"]) then
        sim.setObjectInt32Param(proxFront,sim.objintparam_visibility_layer,1)
        sim.setObjectInt32Param(proxBack,sim.objintparam_visibility_layer,1)
        sim.setObjectInt32Param(proxFrontLeft,sim.objintparam_visibility_layer,1)
        sim.setObjectInt32Param(proxBackLeft,sim.objintparam_visibility_layer,1)
        sim.setObjectInt32Param(proxFrontRight,sim.objintparam_visibility_layer,1)
        sim.setObjectInt32Param(proxBackRight,sim.objintparam_visibility_layer,1)
        sim.setObjectInt32Param(proxFrontLeftLeft,sim.objintparam_visibility_layer,1)
        sim.setObjectInt32Param(proxBackLeftLeft,sim.objintparam_visibility_layer,1)
        sim.setObjectInt32Param(proxBackRightRight,sim.objintparam_visibility_layer,1)
        sim.setObjectInt32Param(proxFrontRightRight,sim.objintparam_visibility_layer,1)
    else
        sim.setObjectInt32Param(proxFront,sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Param(proxBack,sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Param(proxFrontLeft,sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Param(proxBackLeft,sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Param(proxFrontRight,sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Param(proxBackRight,sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Param(proxFrontLeftLeft,sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Param(proxBackLeftLeft,sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Param(proxBackRightRight,sim.objintparam_visibility_layer,0)
        sim.setObjectInt32Param(proxFrontRightRight,sim.objintparam_visibility_layer,0)
    end 
    local sensor_map = {} -- Create a map, that is of the same type as the 
    -- prox_sensor message structure. 
    sensor_map['prox_front_left'] = proxfl
    sensor_map['prox_front'] = proxf
    sensor_map['prox_front_right'] = proxfr
    sensor_map['prox_back'] = proxb
    sensor_map['prox_back_left'] = proxbl
    sensor_map['prox_back_right'] = proxbr
    sensor_map['prox_back_left_left'] = proxbll
    sensor_map['prox_back_right_right'] = proxbrr
    sensor_map['prox_front_left_left'] = proxfll
    sensor_map['prox_front_right_right'] = proxfrr

    -- print(sensor_map)
    simROS.publish(prox_pub, sensor_map)
end



-- #######################################
-- SUBSCRIBERING
-- Callback function for the motors.
-- takes the cmd_vel message and sets the motor velocity based on 
-- that message.
function Motor_control_callback(control)
    d= 0.381/2
    angle = control.steering_angle
    vel = control.speed
    sim.setJointTargetVelocity(motorLeft,2*vel)
    sim.setJointTargetVelocity(motorRight,2*vel)

    -- We handle the front left and right wheel steerings (Ackermann steering):
    steeringAngleLeft=math.atan(l/(-d+l/math.tan(angle)))
    steeringAngleRight=math.atan(l/(d+l/math.tan(angle)))
    sim.setJointTargetPosition(steeringLeft,steeringAngleLeft)
    sim.setJointTargetPosition(steeringRight,steeringAngleRight)
end
--
--####################################
--HELPER FUNCTIONS 
function int_to_bool(val)
    if val == 1 then
        return false
    else
        return true
    end
end

function getTransformStamped(objHandle, name, relTo, relToName)
    t = sim.getSystemTime()
    p = sim.getObjectPosition(objHandle, relTo)
    o = sim.getObjectQuaternion(objHandle, relTo)
    return {
        header = {stamp = t, frame_id = relToName},
        child_frame_id = name,
        transform = {
            translation = {x = p[1], y = p[2], z = p[3]},
            rotation = {x = o[1], y = o[2], z = o[3], w = o[4]}
        }
    }
end




function sysCall_cleanup() 
 if simROS then   
        -- Shut down publisher and subscriber. Not really needed from a simulation script (automatic shutdown)
        simROS.shutdownSubscriber(motor_sub)
        simROS.shutdownPublisher(prox_pub)
    end
end 
