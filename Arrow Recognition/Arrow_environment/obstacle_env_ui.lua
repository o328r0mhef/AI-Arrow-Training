ackermann_path = "models/ROS_robots/ackermann_robot.ttm"
omni_path = "models/ROS_robots/omni_robot.ttm"
dif_path = "models/ROS_robots/differential_robot.ttm"
last_id = 0
function sysCall_init()
    print("init")
    -- do some initialization here
end

function sysCall_nonSimulation()
    --print("test")
    createDlg()
    -- is executed when simulation is not running
end

function sysCall_beforeSimulation()
    print("before")
    -- is executed before a simulation starts
end

function sysCall_afterSimulation()
    print("after")
    -- is executed before a simulation ends
end

function sysCall_cleanup()
    print("celanup")
    -- do some clean-up here
end

function createDlg()
    if not ui then
        local xml = [[
        <ui title="Select Robot" closeable="false" resizable="false" activate="false" >


        <label text="differential robot" style="* {font-weight: bold;}"/>
        <group layout="hbox" flat="true" style="* {margin-top: 0px; margin-bottom: 0px;}">
        <button text="differential robot" checked="false" on-click="switch_robot" id="1" style="* {width: 120px; height: 25px; margin-top: 0px;}"/>
        </group>

        <label text="ackermann robot" style="* {font-weight: bold;}"/>
        <group layout="hbox" flat="true" style="* {margin-top: 0px; margin-bottom: 0px;}">
        <button text="ackermann robot" checked="false" on-click="switch_robot" id="2" style="* {width: 120px; height: 25px; margin-top: 0px;}"/>
        </group>

        <label text="omni wheel robot" style="* {font-weight: bold;}"/>
        <group layout="hbox" flat="true" style="* {margin-top: 0px; margin-bottom: 0px;}">
        <button text="omni wheel robot" checked="false" on-click="switch_robot" id="3" style="* {width: 120px; height: 25px; margin-top: 0px;}"/>
        </group>


        </ui>
        ]]
        ui={}
        ui.dlg=simUI.create(xml)
    end
end


function switch_robot(ui,id,newVal)
    pos = {4.1515,0.00013588,0.072499}
    print("id")
    print(id)
    print("last id")
    print(last_id)
    
    
    if (id == 1 and id ~= last_id) then
    print("differential robot loading")
    simLoadModel(dif_path)
    objectHandle=sim.getObject("/differential_robot")
    sim.setObjectPosition(objectHandle,sim.handle_world,pos)
    elseif id == 2 and id ~= last_id then
    print("ackerman robot loading")
    simLoadModel(ackermann_path)
    objectHandle=sim.getObject("/Ack_robot")
    sim.setObjectPosition(objectHandle,sim.handle_world,pos)
    elseif id == 3 and id ~= last_id then
    print("omni wheel robot loading")
    simLoadModel(omni_path)
    objectHandle=sim.getObject("/Omni_robot")
    sim.setObjectPosition(objectHandle,sim.handle_world,pos)
    end
    
    if last_id ~= id and last_id ~= 0 then
        if last_id == 1 then
            objectHandle=sim.getObject("/differential_robot")
            sim.removeModel(objectHandle)
        elseif last_id == 2 then
            objectHandle=sim.getObject("/Ack_robot")
            sim.removeModel(objectHandle)
        elseif last_id == 3 then
            objectHandle=sim.getObject("/Omni_robot")
            sim.removeModel(objectHandle)   
        end
    end
    
        
    last_id = id
     
end
