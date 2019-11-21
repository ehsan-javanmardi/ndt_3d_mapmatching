# ndt_3d_mapmatching
self localization based on 3D NDT map matching

Basic node to get the location of the car using matching the scan to the map
Ehsan Javanmardi
 2019.11.06
    ndt_3D_mapmatching_v6

    Edited so that we can use it for new velodyne driver
    New velodyne driver does not have  pointy_type.h so I bring it from old code
    it was #include <velodyne_pointcloud/point_types.h>  -->    #include <kmj_point_type/point_types.h>

2017.12.21
    ndt_3D_mapmatching_v5.1
    Option: get groundtruth as initila_guess : (groundtruth_as_prediction)

2017.12.17
    ndt_3D_mapmatching_v5.0
    based on ndt_3D_mapmatching_v4.0

2017.07.14
    ndt_3D_mapmatching_v4.0


