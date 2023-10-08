function sysCall_init()
    maxScanDistance = 5
    showLines = true
    generateData = true
    rangeData = true -- if false, then X/Y/Z data rel to sensor base
    discardMaxDistPts = true

    self = sim.getObject('.')
    visionSensors = {sim.getObject("./sensor1"), sim.getObject("./sensor2")}
    local collection = sim.createCollection(0)
    sim.addItemToCollection(collection, sim.handle_all, -1, 0)
    sim.addItemToCollection(collection, sim.handle_tree, self, 1)
    sim.setObjectInt32Param(visionSensors[1],
                            sim.visionintparam_entity_to_render, collection)
    sim.setObjectInt32Param(visionSensors[2],
                            sim.visionintparam_entity_to_render, collection)
    sim.setObjectFloatParam(visionSensors[1], sim.visionfloatparam_far_clipping,
                            maxScanDistance)
    sim.setObjectFloatParam(visionSensors[2], sim.visionfloatparam_far_clipping,
                            maxScanDistance)
    red = {1, 0, 0}
    lines = sim.addDrawingObject(sim.drawing_lines, 1, 0, -1, 10000, red)

    pub = simROS.advertise('/hokuyo', 'sensor_msgs/LaserScan')
end

function sysCall_cleanup()
    simROS.shutdownPublisher(pub)
    sim.removeDrawingObject(lines)
end

function sysCall_sensing()
    local measuredData = {}
    if (sim.unpackTable(sim.getStringSignal("lidar"))[1] == 2) then
        generateData = true
        print("gen true")
    else
        generateData = false
        print("gen false")
    end


    if (sim.unpackTable(sim.getStringSignal("render_lidar"))[1] == 2) then
        showLines = true
        print("show true")
    else
        showLines = false
        print("show false")
    end

    sim.addDrawingObjectItem(lines, nil)
    for i = 1, 2, 1 do
        local r, t, u = sim.readVisionSensor(visionSensors[i])
        if u then
            local sensorM = sim.getObjectMatrix(visionSensors[i],
                                                sim.handle_world)
            local relRefM = sim.getObjectMatrix(self, sim.handle_world)
            sim.invertMatrix(relRefM)
            relRefM = sim.multiplyMatrices(relRefM, sensorM)
            local p = {0, 0, 0}
            p = sim.multiplyVector(sensorM, p)
            t = {p[1], p[2], p[3], 0, 0, 0}
            for j = 0, u[2] - 1, 1 do
                for k = 0, u[1] - 1, 1 do
                    local w = 2 + 4 * (j * u[1] + k)
                    local v = {u[w + 1], u[w + 2], u[w + 3], u[w + 4]}
                    if generateData then
                        if rangeData then
                            table.insert(measuredData, v[4])
                        else
                            if v[4] < maxScanDistance * 0.9999 or
                                not discardMaxDistPts then
                                p = sim.multiplyVector(relRefM, v)
                                table.insert(measuredData, p[1])
                                table.insert(measuredData, p[2])
                                table.insert(measuredData, p[3])
                            end
                        end
                    end
                    if showLines then
                        p = sim.multiplyVector(sensorM, v)
                        t[4] = p[1]
                        t[5] = p[2]
                        t[6] = p[3]
                        sim.addDrawingObjectItem(lines, t)
                    end
                end
            end
        end
    end
    if #measuredData > 0 then
        local msg = {
            header = {stamp = simROS.getTime(), frame_id = 'Omni_robot'},
            angle_min = -120 * math.pi / 180,
            angle_max = 120 * math.pi / 180,
            angle_increment = (240 / 684) * math.pi / 180, -- angle: 240 deg, pts: 684
            time_increment = 0,
            scan_time = sim.getSimulationTimeStep(),
            range_min = 0.001,
            range_max = maxScanDistance,
            ranges = measuredData,
            intensities = {}
        }
        simROS.publish(pub, msg)
    end
end
