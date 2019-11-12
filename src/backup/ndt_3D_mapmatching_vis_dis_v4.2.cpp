/*


 Localization program using distortion removed visual Normal Distributions Transform
 Matching scan over point cloud map



 Ehsan Javanmardi

 2017.10.23
 ndt_3D_mapmatching_vis_dis_v4.2
    Code was cleaned
    {
        added comments
        initial_scan_pub name is changed to initialguess_scan_pub
        launch file variable name changed
        xxx_startpoint --> xxx_startpoint_deg (only for rotation angles changed)
        gridSize is chganged to ndt_grid_size
        ndt_ added in begining of some of the ndt parameters
    }

    Get ndt_search_radius from launch file
    Separate do_distortion_removal to do_iterative_distortion_removal and do_matching_distortion_removal
        *in the case of iterative, distortion removal is done inside the ndt optimization algorithm
        *in the case of matching, distortion is removed after first ndt_matching and the results is used in next
         ndt_matching which is give us final results. In this case also calculate double align_improvement_after_dis.

    Saved file name with grid size and search radius and downsampling
    Call setSearchResolution only in the main function instead of calling before matching
    Broadcast Vehicle TF to /base_link
    Save results of first alignment which is used for distortion removal
        this is saved so that later we can compare the matching results with distortion and without distortion.
    Save and load predicted pose list







 2017.07.15
 ndt_3D_mapmatching_vis_dis_v4.1

 TETS :
    Code without Distortion removal works well
    But with distortion removal it fails most of the time

    Entropy part has some problems that I didn't correct it yet!!
    For next version I will deal with this (v4.1)


 CHANGE LOG :
    Changes for version : ndt_3D_mapmatching_vis_dis_v4.0
        Based on based on ndt_3D_mapmatching_vis_dis_v3.1
        Use new point type PointXYZIT doe delta_t and distortion
        Point type is defined in self_driving_point_type.h
        Report more evaluation parameters

    Changes for this version
        Entropy problem from v4.0 is solved



    Added Hitogram featurs

 */

#define OUTPUT

#define PCL_NO_PRECOMPILE

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <chrono>

#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/PointCloud2.h>
#include <velodyne_pointcloud/point_types.h>
#include <velodyne_pointcloud/rawdata.h>

#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/voxel_grid.h>
#include <math.h>
#include <boost/filesystem.hpp>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <ndt/ndt_vis_distortion.h>
#include "distortion.h"
#include "entropy.h"
#include "kmj_self_driving_common.h"
#include "self_driving_point_type.h"

#include <iomanip> // setprecision
#include <sstream> // stringstream

using namespace boost::filesystem;

pose previous_pose, current_pose, predict_pose;
pose offset; // current_pos - previous_pose

// Visual NDT with distortion removal
NormalDistributionsTransform_Visual_dis<PointXYZIT, PointXYZIT> ndt;

// Default values
int ndt_iter = 30; // Maximum iterations
double ndt_grid_size = 1.0; // Resolution
double ndt_search_radius = 1.0; // Search radius
double ndt_step_size = 0.1; // Step size
double ndt_trans_eps = 0.01; // Transformation epsilon

// Leaf size of VoxelGrid filter.

double voxel_leaf_size = 1.0;

// publishers

ros::Publisher map_pub;
ros::Publisher ndt_map_pub;

ros::Publisher scan_pub; // raw scan from velodyne
ros::Publisher calibrated_scan_pub; // after calibration of raw scan with sensor parameters
ros::Publisher initialguess_scan_pub; // initial guess scan for begining
ros::Publisher predicted_scan_pub;
ros::Publisher filtered_scan_pub;
ros::Publisher aligned_scan_pub;
ros::Publisher transformed_scan_pub;
ros::Publisher transformed_dis_scan_pub;

ros::Publisher carTrajectory_pub;

// For calculating map information

ros::Publisher origin_dirty_centroids_pub;
ros::Publisher dirty_centroids_pub;
ros::Publisher ground_dirty_centroids_pub;
ros::Publisher nonground_dirty_centroids_pub;
ros::Publisher dirty_centroids_origin_pub;
ros::Publisher map_in_velodyne_range_pub;
ros::Publisher points_x_pub;
ros::Publisher points_y_pub;
ros::Publisher useful_oned_centroids_pub;

// show data in rviz

bool show_scan = true;
bool show_filtered_scan = true;
bool show_transformed_scan = true;
bool show_initial_scan = true;
bool show_map = true;
bool show_car_trajectory = true;
bool show_transformed_dis_scan = true;

bool do_iterative_distortion_removal = false;
bool do_matching_distortion_removal = false;

// save scan data

bool save_transformed_scan = false;
bool save_predicted_scan = false;
bool save_aligned_scan = false;
bool save_transformed_dis_scan = false;
bool save_pre_transformed_scan = false;
bool save_prediction_to_file = false;
std::string save_path = "/home/ehsan/workspace/results/map_matching/";
std::string predicted_pose_path = "/home/ehsan/workspace/input_data/predicted_pose_list.csv";

bool use_prediction_from_file = false;

// map file pathes
std::string map_file_path;
std::string map_file_name;
pcl::PointCloud<PointXYZIT>::Ptr map_ptr (new pcl::PointCloud<PointXYZIT>);

// if 0 then map will loaded by name and if not
int map_load_mode = 0;

// If the map is loaded, map_loaded will be true.
bool map_loaded = false;

// time variables

ros::Time current_scan_time;
ros::Time previous_scan_time;
ros::Duration scan_duration;
std::chrono::time_point<std::chrono::system_clock> \
        matching_start, matching_end, downsample_start, downsample_end, \
        align_start, align_end;
double alignTime, matchingTime, downSampleTime ;

Eigen::Matrix4f tf_predict, tf_previous, tf_current;

int skipSeq;

// start point

double x_startpoint = 0.0;
double y_startpoint = 0.0;
double z_startpoint = 0.0;

double yaw_startpoint =  0.0; // angles are all in radian (-45/180.0) * M_PI;
double roll_startpoint =  0.0; // (0/180.0) * M_PI;
double pitch_startpoint = 0.0; // (-33/180.0) * M_PI;

double velodyne_yaw_offset = 0.0;

double align_improvement_after_dis = -1.0;

// log of localization information

FILE * pFileLog;

pcl::PCDWriter writer;
std::vector<pose> carPoseList;

// for loading predicted pose from file

struct pose_entity
{
    pose _pose;
    int seq;
};

std::vector<pose_entity> predicted_pose_list;




void scan_callback(const sensor_msgs::PointCloud2::ConstPtr& input)
{
    current_scan_time = input->header.stamp;



    // SKIP SPECIFIED NUMBER OF SCAN ########################################################################

    if (input->header.seq < (unsigned int) skipSeq)
    {
        std::cout << "skip " << input->header.seq << std::endl;
        return;
    }

    // CHECK IF MAP IS LOADED OR NOT ########################################################################

    if (!map_loaded)
    {
        std::cout << "map is not loaded......... velodyne seq is : " << input->header.seq << std::endl;
        return;
    }

    // CHECK DISTORTION REMOVAL OPTION IS SET CORRECTLY OR NOT

    if (do_iterative_distortion_removal && do_matching_distortion_removal)
    {
        ROS_ERROR("Both do_iterative_distortion_removal and do_matching_distortion_removal are sset to 1");
        return;
    }



    // SHOW MAP IN RVIZ #####################################################################################

    if (show_map)
    {
        publish_pointCloud(*map_ptr, map_pub, "map");

        pcl::VoxelGridCovariance<PointXYZIT> target_cells;
        ndt.getCells(target_cells);

        //pcl::VoxelGridCovariance<pcl::PointXYZ>::Leaf leaf_;
        //std::map<size_t, pcl::VoxelGridCovariance<pcl::PointXYZI>::Leaf> leaves;
        //leaves = target_cells.getLeaves();
        //std::vector<Leaf> leafList;
        //getLeaves(leafList);

        typedef pcl::VoxelGridCovariance<PointXYZIT> VectorCovarianceXYZ;
        typedef typename VectorCovarianceXYZ::Leaf VectorCovarianceLeaf;
        typedef std::vector<VectorCovarianceLeaf> VectorCovarianceLeafList;

        VectorCovarianceLeafList leafList;
        ndt.getLeaves(leafList);

        visualization_msgs::MarkerArray ndtSphereList;
        //Eigen::Vector4d RGBA(0.35, 0.7, 0.8, 0.2);
        Eigen::Vector4d normalDistribution_color(0.35, 0.7, 0.8, 0.4);

        //double d1 = ndt.get_d1();
        //double d2 = ndt.get_d2();

        // a 90% confidence interval corresponds to scale=4.605
        //showCovariance(leaves, ndtSphereList, 4.605 ,"map", RGBA, d1, d2);
        //showCovariance(leaves, ndtSphereList, 4.605 ,"map", normalDistribution_color);
        setCovarianceListMarker<pcl::VoxelGridCovariance<PointXYZIT>::Leaf>(leafList, ndtSphereList, \
                                                      4.605 ,"map", 20);

        setCovarianceListMarker<pcl::VoxelGridCovariance<PointXYZIT>::Leaf>(leafList, ndtSphereList, \
                                                      4.605 ,"map", normalDistribution_color, 20);

        ndt_map_pub.publish(ndtSphereList);

        show_map = 0;
    }

    // CONVERT MESSSAGE TO POINT CLOUD ######################################################################

    pcl::PointCloud<velodyne_pointcloud::PointXYZIR> scan_xyzir;
    pcl::fromROSMsg(*input, scan_xyzir);

    pcl::PointCloud<velodyne_pointcloud::PointXYZIR> calibrated_scan_xyzir;

    // CALIBRATE POINT CLOUD SUCH THAT THE SENSOR BECOME PERPENDICULAR TO GROUND SURFACE ####################
    // THIS STEP HELPS TO REMOVE GROUND SURFACE MUCH EASIER

    pose pose_lidar(0.0, 0.0, 0.0, roll_startpoint, pitch_startpoint, 0.0);
    Eigen::Matrix4f tf_lidar;

    pose_to_tf(pose_lidar, tf_lidar);

    pcl::transformPointCloud(scan_xyzir, calibrated_scan_xyzir, tf_lidar);

    publish_pointCloud(calibrated_scan_xyzir, calibrated_scan_pub, "/velodyne");

    // DO NOT REMOVE GROUND #################################################################################

    pcl::PointCloud<PointXYZIT> scan;

    for (pcl::PointCloud<velodyne_pointcloud::PointXYZIR>::const_iterator item = calibrated_scan_xyzir.begin(); \
         item != calibrated_scan_xyzir.end(); item++)
    {
        PointXYZIT p;

        p.x = (double) item->x;
        p.y = (double) item->y;
        p.z = (double) item->z;
        p.intensity = (double) item->intensity;
        //p.ring = item->ring;

        if (getR(p) > 1.0 && getR(p) < 90.0)
            scan.points.push_back(p);
    }

    // STATIC GROUND AND CAR ROOF REMOVAL ###################################################################
/*
    pcl::PointCloud<pcl::PointXYZI> scan;

    for (pcl::PointCloud<velodyne_pointcloud::PointXYZIR>::const_iterator item = calibrated_scan_xyzir->begin(); \
         item != calibrated_scan_xyzir->end(); item++)
    {

        //if (item->z > -1.8 && item->z < 2.0)
        if (item->z > -1.5 && item->z < 2.0)
        {
            //if the points placed in the roof of vehicle
            if (item->x > 0.5 && item->x < 2.2 && item->y < 0.8 && item->y > -0.8);
            else
            {
                pcl::PointXYZI p;

                p.x = (double) item->x;
                p.y = (double) item->y;
                p.z = 0.0; // because 3D matching
                p.intensity = (double) item->intensity;
                //p.ring = item->ring;

                scan3D_ptr->points.push_back(p);
            }
        }
    }
*/
    publish_pointCloud(scan, scan_pub, "/velodyne");

    // TRANSFORM SCAN TO GLOBAL COORDINATE IT FROM LOCAL TO GLOBAL ##########################################

    int GPS_enabled = 0;

    if (GPS_enabled)
    {
/*
        // get current locaition from GPS
        // translate scan to current location

        // get x,y,z and yaw from GPS
        // this will be predict_pose

        std::cout << "##### use GPS !!" << std::endl;

        pose gps;


        Eigen::Translation3f predict_translation(gps.x, gps.y, gps.z);
        Eigen::AngleAxisf predict_rotation_x(roll_startpoint , Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf predict_rotation_y(pitch_startpoint, Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf predict_rotation_z(gps.yaw, Eigen::Vector3f::UnitZ());

        Eigen::Matrix4f tf_predict = (predict_translation * predict_rotation_z * predict_rotation_y * predict_rotation_x).matrix();
*/

    }
    else
    {
        // local to global using tf_ltob x,y,z,yaw(heading)
        // calibrate
        // estimate current position using previous and offset
        // Guess the initial gross estimation of the transformation

        offset.roll = 0.0;
        offset.pitch = 0.0;
        predict_pose = previous_pose + offset;

        pose_to_tf(predict_pose, tf_predict);
        pose_to_tf(previous_pose, tf_previous);
    }

    // SAVE PREDICTION TO FILE ##############################################################################

    if (save_prediction_to_file)
    {
        FILE *pFile;
        std::string name = save_path + "predicted_pose_list.csv";

        pFile = fopen(name.c_str(), "a");

        pose temp_pose;
        tf_to_pose(tf_predict, temp_pose);

        fprintf(pFile, "%i,%f,%f,%f,%f,%f,%f\n", input->header.seq, temp_pose.x, temp_pose.y, temp_pose.z, \
                temp_pose.roll, temp_pose.pitch, temp_pose.yaw);

        fclose(pFile);

    }

    // EXTRACT CURRENT PREDICTION FROM PREDICTED POSE LIST ##################################################

    if (use_prediction_from_file)
    {

        std::vector<pose_entity>::const_iterator item = predicted_pose_list.begin();

        int scan_seq = input->header.seq;

        if (scan_seq < item->seq)
        {
            ROS_WARN("scan with seq %i skipped to align the predicted file and scans");
            return;
        }
        else
        {
            while (item != predicted_pose_list.end() && item->seq < scan_seq)
            {
                item++;
            }

            if (item->seq != scan_seq)
            {
                ROS_ERROR("scan seq and predicted pose seq are not same. %i != %i", scan_seq, item->seq);
                // Do nothing
                return;
            }
            else
            {
                pose pose_from_file;
                pose_from_file = item->_pose;
                pose_to_tf(pose_from_file, tf_predict);

                std::cout << "Use prediction list for " << item->seq << std::endl;
            }
        }
    }


    // SHOW PREDICTED SCAN ##################################################################################

    if (show_scan)
    {
        pcl::PointCloud<PointXYZIT>  predicted_scan;

        pcl::transformPointCloud(scan, predicted_scan, tf_predict);

        publish_pointCloud(predicted_scan, predicted_scan_pub, "map");

        // SHOW CALIBRATED INITIAL POSE IN GLOBAL COORDINATE

        if (show_initial_scan)
        {
            publish_pointCloud(predicted_scan, initialguess_scan_pub, "map");
            show_initial_scan = false;
        }
    }

    // SAVE PREDICTED CALIBRATED SCAN ##################################################

    if (save_predicted_scan)
    {
        pcl::PointCloud<PointXYZIT> predicted_scan;

        pcl::transformPointCloud(scan, predicted_scan, tf_predict);

        predicted_scan.height = 1;
        predicted_scan.width = predicted_scan.size();
        predicted_scan.points.resize (predicted_scan.width * predicted_scan.height);

        std::string name = save_path + "predicted_scan/predicted_scan_" + \
                           std::to_string((input->header.seq)) + ".pcd";

        writer.write(name, predicted_scan, false);
    }

    // UPDATE DELTA T FOR THE SCAN ##########################################################################

    if (do_iterative_distortion_removal)
        calculateDeltaT(scan, scan);

    // DOWNSAMPLE SCAN USING VOXELGRID FILTER ###############################################################

    downsample_start =  std::chrono::system_clock::now();

    pcl::PointCloud<PointXYZIT>::Ptr input_cloud_ptr(new pcl::PointCloud<PointXYZIT>(scan));
    pcl::PointCloud<PointXYZIT> filtered_scan;

    pcl::ApproximateVoxelGrid<PointXYZIT> voxel_grid_filter;
    voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
    voxel_grid_filter.setInputCloud(input_cloud_ptr);
    voxel_grid_filter.filter(filtered_scan);

    downsample_end =  std::chrono::system_clock::now();

    downSampleTime = std::chrono::duration_cast<std::chrono::microseconds>\
                                        (downsample_end - downsample_start).count()/1000.0;

    // ALIGN POINT CLOUD TO THE MAP #########################################################################

    align_start = std::chrono::system_clock::now();

    //pcl::PointCloud<pcl::PointXYZI>::Ptr scan_ptr(new pcl::PointCloud<pcl::PointXYZI>(scan));

    pcl::PointCloud<PointXYZIT>::Ptr filter_scan_ptr(new pcl::PointCloud<PointXYZIT>(filtered_scan));

    ndt.setInputSource(filter_scan_ptr);

    if (do_iterative_distortion_removal)
        ndt.setPreviousTF(tf_previous);

    //ndt.setSearchResolution(ndt_search_radius);

    pcl::PointCloud<PointXYZIT> aligned_scan;

    // aligned scan is distortion removed downsampled scan so it is not useful

    ndt.align(aligned_scan, tf_predict);

    // get the global translation matrix

    Eigen::Matrix4f tf_align(Eigen::Matrix4f::Identity());
    tf_align = ndt.getFinalTransformation();


    //  DO DISTORTION ELIMINATION AND MATCH ONE MORE TIME ###################################################

    if (do_matching_distortion_removal)
    {


        // MAKE ONLY TRANSFORMED SCAN (DISTORTION NOT REMOVED) ##############################################

        pcl::PointCloud<PointXYZIT> pre_transformed_scan;

        pcl::transformPointCloud(scan, pre_transformed_scan, tf_align);

        pre_transformed_scan.height = 1;
        pre_transformed_scan.width = pre_transformed_scan.size();
        pre_transformed_scan.points.resize (pre_transformed_scan.width * pre_transformed_scan.height);


        // SAVE PRE TRANSFORMED SCAN ########################################################################

        if (save_pre_transformed_scan)
        {
            std::string name = save_path + "pre_transformed_scan/pre_transformed_scan_" + \
                               std::to_string((input->header.seq)) + ".pcd";

            writer.write(name, pre_transformed_scan, false);

            if (scan.size() != pre_transformed_scan.size())
            {
                PCL_ERROR("number of scan and transformed_scan are not same");
            }

        }



        Eigen::Matrix4f tf_align_pre = tf_align;

        // remove the distortion according to first matching result
        pcl::PointCloud<PointXYZIT> dis_eliminated_scan;

        std::vector<double> delta_t;

        calculateDeltaT(scan, delta_t);

        removeDistortion(scan, dis_eliminated_scan, tf_previous, tf_align, delta_t);

        // down-sample distortion eliminated scan

        pcl::PointCloud<PointXYZIT>::Ptr input_cloud_ptr(new pcl::PointCloud<PointXYZIT>(dis_eliminated_scan));
        pcl::PointCloud<PointXYZIT> filtered_scan;

        pcl::ApproximateVoxelGrid<PointXYZIT> voxel_grid_filter;
        voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
        voxel_grid_filter.setInputCloud(input_cloud_ptr);
        voxel_grid_filter.filter(filtered_scan);

        // perform ndt_scan matching one more time

        pcl::PointCloud<PointXYZIT>::Ptr filter_scan_ptr(new pcl::PointCloud<PointXYZIT>(filtered_scan));

        ndt.setInputSource(filter_scan_ptr);

        //ndt.setSearchResolution(ndt_search_radius);

        pcl::PointCloud<PointXYZIT> aligned_scan;

        ndt.align(aligned_scan, tf_predict);

        // get the global translation matrix

        Eigen::Matrix4f tf_align(Eigen::Matrix4f::Identity());
        tf_align = ndt.getFinalTransformation();

        // calculate the difference of scan matching result without distortion removal and with distortion removal

        double dx, dy, dz;

        dx = tf_align_pre(0,3) - tf_align(0,3) ;
        dy = tf_align_pre(1,3) - tf_align(1,3) ;
        dz = tf_align_pre(2,3) - tf_align(2,3) ;

        align_improvement_after_dis = sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2));
    }

    tf_to_pose(tf_align, current_pose);

    int iteration = ndt.getFinalNumIteration();
    // double score = ndt.getScore();
    double trans_probability = ndt.getTransformationProbability();
    int input_point_size = filter_scan_ptr->size();

    offset = current_pose - previous_pose;

    double displacement_3d = sqrt(pow(offset.x, 2.0) + pow(offset.y, 2.0) + pow(offset.z, 2.0));

    pose initial_guess_error_pose;

    initial_guess_error_pose = current_pose - predict_pose;

    double initial_guess_yaw_error = (fabs(predict_pose.yaw - current_pose.yaw)) / M_PI * 180.0;

    double initial_guess_error = sqrt(pow(initial_guess_error_pose.x, 2.0) + \
                                      pow(initial_guess_error_pose.y, 2.0) + \
                                      pow(initial_guess_error_pose.z, 2.0));

    previous_pose = current_pose;

    carPoseList.push_back(current_pose);

    align_end = std::chrono::system_clock::now();

    alignTime = std::chrono::duration_cast<std::chrono::microseconds>\
                                        (align_end - align_start).count()/1000.0;

    matchingTime = alignTime + downSampleTime;


    // BROADCAST VEHICLE TF SO THAT WE CAN SEE THE DRIVING VIEW IN RVIZ #####################################

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion current_q;

    current_q.setRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);

    transform.setOrigin(tf::Vector3(current_pose.x, current_pose.y, current_pose.z));
    transform.setRotation(current_q);

    ros::Time current_scan_time;

    current_scan_time = input->header.stamp;

    br.sendTransform(tf::StampedTransform(transform, current_scan_time, "/map", "/base_link"));



    // CALCULATE ENTROPY RELATED PARAMETERS #################################################################



    bool do_entropy_calculation = true;
    bool do_dshape_calculation = true;
    bool do_layout_calculation = true;
    bool do_xy_points_entropy = true;

    if (do_entropy_calculation)
    {
        pcl::VoxelGridCovariance<PointXYZIT> target_cells;

        ndt.getCellsNew(target_cells);

        std::vector<Eigen::Vector3d> dirty_centroids_Dimensions_values;
        pcl::PointCloud<PointXYZINormal> dirty_centroids;
        std::vector<int> dirty_centroids_Dimensions;

        // D_shape << D1, D2, D3;

        getNearestDirtyCentroidsWithNormal(target_cells, filtered_scan, dirty_centroids,\
                                           dirty_centroids_Dimensions_values,\
                                           dirty_centroids_Dimensions,\
                                           tf_align, ndt_search_radius);

        publish_pointCloud(dirty_centroids, dirty_centroids_pub, "map");

        pcl::PointCloud<pcl::PointXYZINormal> ground_dirty_centroids;
        pcl::PointCloud<pcl::PointXYZINormal> nonground_dirty_centroids;
        pcl::PointCloud<pcl::PointXYZINormal> useful_oned_centroids;
        pcl::PointCloud<pcl::PointXYZINormal> useful_dirty_centroids;

        pcl::PointCloud<pcl::PointXYZINormal> points_x;
        pcl::PointCloud<pcl::PointXYZINormal> points_y;

        std::vector<Eigen::Vector3d> nonground_dirty_centroids_dimensions_values;
        std::vector<int> nonground_dirty_centroids_dimensions;

        for (int i=0; i<dirty_centroids.size(); i++ )
        {
            pcl::PointXYZINormal p = dirty_centroids[i];
            Eigen::Vector3d v = dirty_centroids_Dimensions_values[i];
            int shape = dirty_centroids_Dimensions[i];

            /*if ( shape == 2 )
            {
                if (fabs(p.normal_z) >= fabs(p.normal_x) && fabs(p.normal_z) >= fabs(p.normal_y))
                    ground_dirty_centroids.push_back(p);
                else
                {
                    if (fabs(p.normal_x) >= fabs(p.normal_z) && fabs(p.normal_x) >= fabs(p.normal_y))
                        points_x.push_back(p);
                    else if (fabs(p.normal_y) >= fabs(p.normal_z) && fabs(p.normal_y) >= fabs(p.normal_x))
                        points_y.push_back(p);

                    nonground_dirty_centroids.push_back(p);
                    nonground_dirty_centroids_dimensions_values.push_back(v);
                    nonground_dirty_centroids_dimensions.push_back(shape);
                }

            }
            else if (shape == 1)
            {
                if (fabs(p.normal_z) < fabs(p.normal_x) && fabs(p.normal_z) < fabs(p.normal_y))
                {
                    useful_oned_centroids.push_back(p);
                }
            }
            else if (shape ==3)
            {
            }*/

            if ( shape == 2 && fabs(p.normal_z) >= fabs(p.normal_x) && fabs(p.normal_z) >= fabs(p.normal_y))
                ground_dirty_centroids.push_back(p);
            else
            {
                nonground_dirty_centroids.push_back(p);
                nonground_dirty_centroids_dimensions_values.push_back(v);
                nonground_dirty_centroids_dimensions.push_back(shape);
            }

        }

        //publish_pointCloud(useful_oned_centroids, useful_oned_centroids_pub, "map");
        publish_pointCloud(ground_dirty_centroids, ground_dirty_centroids_pub, "map");
        publish_pointCloud(nonground_dirty_centroids, nonground_dirty_centroids_pub, "map");
        //publish_pointCloud(points_x, points_x_pub, "map");
        //publish_pointCloud(points_y, points_y_pub, "map");

        if (do_dshape_calculation)
        {
            int D1_count, D2_count, D3_count, useful_D1_count;
            D1_count = D2_count = D3_count = useful_D1_count = 0;

            double sum_D1, sum_D2, sum_D3;
            sum_D1 = sum_D2 = sum_D3 = 0.0;

            for (int i=0; i< nonground_dirty_centroids.size(); i++)
            {
                Eigen::Vector3d d_vector = nonground_dirty_centroids_dimensions_values[i];
                int shape = nonground_dirty_centroids_dimensions[i];
                pcl::PointXYZINormal p = nonground_dirty_centroids[i];

                sum_D1 += d_vector(0);
                sum_D2 += d_vector(1);
                sum_D3 += d_vector(2);

                if (shape == 1)
                    ++D1_count;

                if (shape == 2)
                    ++D2_count;

                if (shape == 3)
                    ++D3_count;

                if (shape == 1)
                    if (fabs(p.normal_z) < fabs(p.normal_x) && fabs(p.normal_z) < fabs(p.normal_y))
                        ++useful_D1_count;
            }

            FILE* pFile;

            std::string name = save_path + "nonground_dirty_centroids_dimensions_values.csv";

            pFile = fopen(name.c_str(), "a");

            fprintf (pFile, "%i,", (int)input->header.seq);

            int size_dirty_centroids = nonground_dirty_centroids_dimensions_values.size();

            fprintf (pFile, "%i,", size_dirty_centroids);

            fprintf (pFile, "%i,", D1_count);
            fprintf (pFile, "%i,", D2_count);
            fprintf (pFile, "%i,", D3_count);
            fprintf (pFile, "%i,", useful_D1_count);

            fprintf (pFile, "%f,", (float)(sum_D1 / size_dirty_centroids));
            fprintf (pFile, "%f,", (float)(sum_D2 / size_dirty_centroids));
            fprintf (pFile, "%f\n", (float)(sum_D3 / size_dirty_centroids));

            fclose(pFile);
        }


        if (do_layout_calculation)
        {

            // Make layout

            double layout_non_ground_entropy= 0.0;

            int num_bin = 36;
            int histogram[num_bin]={};

            double min_bin_val = 0.0;
            double max_bin_val = 360.0;

            if (makeLayoutHistogram(nonground_dirty_centroids, histogram,\
                                    num_bin, min_bin_val, max_bin_val, \
                                    tf_align) != 0)
            {
                PCL_ERROR("ERROR in FUNCTION makeLayoutHistogram\n");
            }

            FILE* pFile;

            std::string name = save_path + "layout_histogram_nonground.csv";

            pFile = fopen(name.c_str(), "a");

            fprintf (pFile, "%i,", (int)input->header.seq);
            fprintf (pFile, "%i,", dirty_centroids.size());

            for (int i=0; i<num_bin; i++)
            {
                fprintf (pFile, "%i,", histogram[i]);
            }

            calculateEntropyFromHistogram(histogram, num_bin, layout_non_ground_entropy);

            fprintf (pFile, "%f\n", layout_non_ground_entropy);

            fclose(pFile);

        }

        // calculate layout entropy for only 2D and useful 1D

        if (do_layout_calculation)
        {

            // Make layout

            double layout_non_ground_entropy= 0.0;

            int num_bin = 36;
            int histogram[num_bin]={};

            double min_bin_val = 0.0;
            double max_bin_val = 360.0;

            // only use 2D and useful 1D ndt in layout entropy

            pcl::PointCloud<pcl::PointXYZINormal> nonground_1D_2D_dirty_centroids;

            for (int i=0; i< nonground_dirty_centroids_dimensions.size(); i++)
            {
                if (nonground_dirty_centroids_dimensions[i] == 2 ||\
                    nonground_dirty_centroids_dimensions[i] == 1  )
                {
                    nonground_1D_2D_dirty_centroids.push_back(nonground_dirty_centroids[i]);
                }

            }


            if (makeLayoutHistogram(nonground_1D_2D_dirty_centroids, histogram,\
                                    num_bin, min_bin_val, max_bin_val, \
                                    tf_align) != 0)
            {
                PCL_ERROR("ERROR in FUNCTION makeLayoutHistogram\n");
            }

            FILE* pFile;

            std::string name = save_path + "1D_2D_layout_histogram_nonground.csv";

            pFile = fopen(name.c_str(), "a");

            fprintf (pFile, "%i,", (int)input->header.seq);
            fprintf (pFile, "%i,", dirty_centroids.size());

            for (int i=0; i<num_bin; i++)
            {
                fprintf (pFile, "%i,", histogram[i]);
            }

            calculateEntropyFromHistogram(histogram, num_bin, layout_non_ground_entropy);

            fprintf (pFile, "%f\n", layout_non_ground_entropy);

            fclose(pFile);

        }


        if (do_xy_points_entropy)
        {
            pcl::PointCloud<pcl::PointXYZINormal> origin_nonground_dirty_centroids;

            transformPointCloudToOrigin(nonground_dirty_centroids, origin_nonground_dirty_centroids,\
                                        velodyne_yaw_offset, tf_align);

            publish_pointCloud(origin_nonground_dirty_centroids, dirty_centroids_origin_pub,"velodyne_origin");

            int num_bin = 200;
            int x_histogram[num_bin]={};
            int y_histogram[num_bin]={};
            double min_bin_val = -100.0;
            double max_bin_val = 100.0;

            makeXYCentroidsHistogram(origin_nonground_dirty_centroids, x_histogram, y_histogram,
                                  num_bin, min_bin_val, max_bin_val);

            // save histograms

            std::string name = save_path + "x_points_histogram.csv";

            FILE* pFile;
            pFile = fopen(name.c_str(), "a");

            fprintf (pFile, "%i,", (int)input->header.seq);


            for (int i=0; i<num_bin; i++)
            {
                fprintf (pFile, "%i,", x_histogram[i]);
            }

            double x_points_feature_entropy = 0;

            calculateEntropyFromHistogram(x_histogram, num_bin, x_points_feature_entropy);

            fprintf (pFile, "%f\n", x_points_feature_entropy);

            fclose(pFile);



            name = save_path + "y_points_histogram.csv";

            pFile = fopen(name.c_str(), "a");

            fprintf (pFile, "%i,", (int)input->header.seq);

            for (int i=0; i<num_bin; i++)
            {
                fprintf (pFile, "%i,", y_histogram[i]);
            }

            double y_points_feature_entropy = 0.0;

            calculateEntropyFromHistogram(y_histogram, num_bin, y_points_feature_entropy);

            fprintf (pFile, "%f\n", y_points_feature_entropy);

            fclose(pFile);
        }

        if(true)
        {
            int num_bin = 20;
            int histogram[num_bin]={};
            int oned_count = 0;

            if (makeYawAndNDTNormalDiffHistogram(nonground_dirty_centroids,
                                                 nonground_dirty_centroids_dimensions,
                                                 histogram, oned_count,
                                                 num_bin, tf_align) != 0)
            {
                PCL_ERROR("ERROR in FUNCTION makeYawAndNDTNormalDiffHistogram\n");
            }

            // save histogram

            std::string name = save_path + "yaw_and_NDT_dif_histogram.csv";

            FILE* pFile;

            pFile = fopen(name.c_str(), "a");

            fprintf (pFile, "%i,", (int)input->header.seq);
            fprintf (pFile, "%i,", oned_count);

            for (int i=0; i<num_bin; i++)
            {
                fprintf (pFile, "%i,", histogram[i]);
            }

             double yaw_ndt_normal_diff_entropy = 0.0;

            calculateEntropyFromHistogram(histogram, num_bin, yaw_ndt_normal_diff_entropy);

            fprintf (pFile, "%f\n", yaw_ndt_normal_diff_entropy);

            fclose(pFile);
        }

        if (false)
        {
            // calculate normal histogram and entropy for all points in the range of velodyne


            pcl::PointCloud<PointXYZIT> neighbors_points;
            pcl::KdTreeFLANN<PointXYZIT> kdtree;

            pcl::PointCloud<PointXYZIT> map = *map_ptr;

            pcl::PointCloud<PointXYZIT>::Ptr map_ptr_(new pcl::PointCloud<PointXYZIT> (map));

            kdtree.setInputCloud (map_ptr_);

            PointXYZIT center_point;

            center_point.x = tf_align(0,3);
            center_point.y = tf_align(1,3);
            center_point.z = tf_align(2,3);

            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;

            double velodyne_range = 110.0;

            if ( kdtree.radiusSearch (center_point, velodyne_range, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
            {
                PointXYZIT p;

                for (int i=0; i< pointIdxRadiusSearch.size(); i++)
                {
                    p = map[pointIdxRadiusSearch[i]];
                    neighbors_points.push_back(p);
                }
            }
            else
            {
                PCL_WARN("getTargetPointsInNeiborhood No points found near tf_guess\n" );
            }

            pcl::PointCloud<PointXYZIT>::Ptr neighbors_points_ptr(new pcl::PointCloud<PointXYZIT> (neighbors_points));

            kdtree.setInputCloud (neighbors_points_ptr);

            pcl::PointCloud<PointXYZIT> neighbors_points_velodyne_view;

            pointIdxRadiusSearch.clear();

            for (int i=0; i< aligned_scan.size(); i++)
            {
                if ( kdtree.radiusSearch (aligned_scan[i], 1.0, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
                {
                    PointXYZIT p;

                    for (int i=0; i< pointIdxRadiusSearch.size(); i++)
                    {
                        p = neighbors_points[pointIdxRadiusSearch[i]];
                        neighbors_points_velodyne_view.push_back(p);
                    }
                }

            }

            publish_pointCloud(neighbors_points_velodyne_view, map_in_velodyne_range_pub, "map");

        }

    }



    // SHOW RESULTS ON THE SCREEN ##########################################################

    std::cout << "##############     sequence " << input->header.seq << "    ##############" << std::endl;
    std::cout << "X : " << current_pose.x << std::endl;
    std::cout << "Y : " << current_pose.y << std::endl;
    std::cout << "X : " << current_pose.z << std::endl;
    std::cout << "pitch : " << current_pose.pitch << std::endl;
    std::cout << "Roll : " << current_pose.roll << std::endl;
    std::cout << "yaw : " << current_pose.yaw << std::endl;

    std::cout << "DownSampleTime : " << downSampleTime << std::endl;
    std::cout << "AlignTime : " << alignTime << std::endl;
    std::cout << "MatchingTime : " << matchingTime << std::endl;
    std::cout << "Number of iteration : " << ndt.getFinalNumIteration() << std::endl;
    std::cout << "Number of centroids : " << ndt.getCentroidsCount() << std::endl << std::endl;
    //std::cout << "Score : " << score << std::endl;
    std::cout << "trans_probability : " << trans_probability << std::endl;
    std::cout << "Size of input points after downsampling : " << input_point_size << std::endl;

    std::cout << "3D Displacement (current - previous) in meter " << displacement_3d << std::endl;
    std::cout << "Error of initial guess (3D) in meter " << initial_guess_error << std::endl;
    std::cout << "Yaw error of initial guess in degree " << initial_guess_yaw_error << std::endl;

    std::cout << "Alignment improvement after distortion removal is : " << align_improvement_after_dis << std::endl;


    // save results on the file


    // SHOW CAR TRAJECTORY ####################################################

    /*if (show_car_trajectory)
    {
        visualization_msgs::Marker carTrajectoryLineList;
        Eigen::Vector4d RGBALine(0.2, 0.8, 0.2, 1.0);

        getcarTrajectotyList(carPoseList, score, iteration, carTrajectoryLineList, "map", RGBALine, 50);
        carTrajectory_pub.publish(carTrajectoryLineList);
    }*/

    // SHOW DOWNSAMPLED SCAN ################################################################################

    if (show_filtered_scan)
    {
        pcl::transformPointCloud(filtered_scan, filtered_scan, tf_align);
        publish_pointCloud(filtered_scan, filtered_scan_pub, "map");
    }

    // SHOW ALIGNED SCAN ####################################################################################

    publish_pointCloud(aligned_scan, aligned_scan_pub, "map");

    // SAVE ALIGNED SCAN ######################################################

    if (save_aligned_scan)
    {
        // if iterative distortion elimination is set, this cloud is distortion eliminated
        // this cloud is also downsampled scan

        aligned_scan.height = 1;
        aligned_scan.width = aligned_scan.size();
        aligned_scan.points.resize (aligned_scan.width * aligned_scan.height);

        std::string name = save_path + "aligned_scan/aligned_scan_" + \
                           std::to_string((input->header.seq)) + ".pcd";

        writer.write(name, aligned_scan, false);

        if (filtered_scan.size() != aligned_scan.size())
        {
            PCL_ERROR("number of filtered_scan and aligned_scan are not same");
        }
    }

    // MAKE DISTORTION REMOVED TRANSFORMED SCAN #############################################################

    if (do_matching_distortion_removal || do_iterative_distortion_removal)
    {

        pcl::PointCloud<PointXYZIT> transformed_dis_scan;

        std::vector<double> delta_t;

        calculateDeltaT(scan, delta_t);

        removeDistortion(scan, transformed_dis_scan, tf_previous, tf_align, delta_t);
        pcl::transformPointCloud(transformed_dis_scan, transformed_dis_scan, tf_align);

        // SHOW DISTORTION REMOVED TRANSFORMED SCAN ################################################################################

        if (show_transformed_dis_scan)
        {
            publish_pointCloud(transformed_dis_scan, transformed_dis_scan_pub, "map");
        }

        // SAVE DISTORTION REMOVED TRANSFORMED SCAN #################################################################################

        if (save_transformed_dis_scan)
        {
            std::string name = save_path + "transformed_dis_scan/transformed_dis_scan_" + \
                               std::to_string((input->header.seq)) + ".pcd";

            writer.write(name, transformed_dis_scan, false);

            if (scan.size() != transformed_dis_scan.size())
            {
                PCL_ERROR("number of scan and transformed_dis_scan are not same");
            }
        }
    }

    // MAKE ONLY TRANSFORMED SCAN (DISTORTION NOT REMOVED) #####################################################################

    pcl::PointCloud<PointXYZIT> transformed_scan;

    pcl::transformPointCloud(scan, transformed_scan, tf_align);

    transformed_scan.height = 1;
    transformed_scan.width = transformed_scan.size();
    transformed_scan.points.resize (transformed_scan.width * transformed_scan.height);

    // SHOW ONLY TRANSFORMED SCAN ################################################################################

    if (show_transformed_scan)
    {
        publish_pointCloud(transformed_scan, transformed_scan_pub, "map");
    }

    // SAVE ONLY TRANSFORMED SCAN #################################################################################

    if (save_transformed_scan)
    {
        std::string name = save_path + "transformed_scan/transformed_scan_" + \
                           std::to_string((input->header.seq)) + ".pcd";

        writer.write(name, transformed_scan, false);

        if (scan.size() != transformed_scan.size())
        {
            PCL_ERROR("number of scan and transformed_scan are not same");
        }

    }


    // SAVE LOG FILES ###############################################################

    if (true)
    {

        double dx = tf_predict(0,3) - tf_align(0,3) ;
        double dy = tf_predict(1,3) - tf_align(1,3) ;
        double dz = tf_predict(2,3) - tf_align(2,3) ;


        initial_guess_error = sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2));

        initial_guess_yaw_error = (fabs (predict_pose.yaw - current_pose.yaw) )/ M_PI * 180.0;

        dx = tf_previous(0,3) - tf_align(0,3) ;
        dy = tf_previous(1,3) - tf_align(1,3) ;
        dz = tf_previous(2,3) - tf_align(2,3) ;

        double vehicle_displacement = sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2));

        std::string name = save_path + "position_log.csv";

        FILE* pFile;
        pFile = fopen(name.c_str(), "a");

        //fprintf (pFile, "%i,%f,%f,%f,%f,%f,%f,%i,%f,%f,%i,%f,%f,%f,%f,%i,%i,%f,%f,%f,%f,%f,%f,%f\n",


        fprintf (pFile, "%i,", (int)input->header.seq);

        fprintf (pFile, "%f,", (float)current_pose.x );
        fprintf (pFile, "%f,", (float)current_pose.y );
        fprintf (pFile, "%f,", (float)current_pose.z );
        fprintf (pFile, "%f,", (float)current_pose.roll );
        fprintf (pFile, "%f,", (float)current_pose.pitch );
        fprintf (pFile, "%f,", (float)current_pose.yaw );

        fprintf (pFile, "%i,", (int)ndt.hasConverged() );
        fprintf (pFile, "%i,", (int)ndt.getFinalNumIteration());
        fprintf (pFile, "%f,", (float)ndt.getTransformationProbability());
        fprintf (pFile, "%f,", (float)matchingTime);
        fprintf (pFile, "%i,", (int)scan.size());
        fprintf (pFile, "%i,", (int)filtered_scan.size());
        fprintf (pFile, "%f,", (float)initial_guess_error);
        fprintf (pFile, "%f,", (float)initial_guess_yaw_error); // indegree
        fprintf (pFile, "%f,", (float)vehicle_displacement);
        fprintf (pFile, "%f,", (float)align_improvement_after_dis);
        fprintf (pFile, "\n");

        fclose(pFile);

    }


}

int main(int argc, char **argv)
{
    std::cout << "ndt_3D_mapmatching_vis_dis_v4_2\n" ;
    ros::init(argc, argv, "ndt_3D_mapmatching_vis_dis_v4_2");


    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");



    // skip sequence from beginning

    skipSeq = 0;

    if (private_nh.getParam("skipSeq", skipSeq) == false)
    {
      std::cout << "skipSeq is not set." << std::endl;
      //return -1;
    }
    std::cout << "skipSeq: " << skipSeq << std::endl;



    // start point

    if (private_nh.getParam("x_startpoint", x_startpoint) == false)
    {
        std::cout << "x_startpoint is not set." << std::endl;
        //return -1;
    }
    std::cout << "x_startpoint: " << x_startpoint << std::endl;

    if (private_nh.getParam("y_startpoint", y_startpoint) == false){
        std::cout << "y_startpoint is not set." << std::endl;
        //return -1;
    }
    std::cout << "y_startpoint: " << y_startpoint << std::endl;

    if (private_nh.getParam("z_startpoint", z_startpoint) == false)
    {
        std::cout << "z_startpoint is not set." << std::endl;
        //return -1;
    }
    std::cout << "z_startpoint: " << z_startpoint << std::endl;

    // yaw_startpoint is in degree

    double yaw_startpoint_deg, pitch_startpoint_deg, roll_startpoint_deg = 0.0;

    if (private_nh.getParam("yaw_startpoint_deg", yaw_startpoint_deg) == false)
    {
        std::cout << "yaw_startpoint_deg is not set." << std::endl;
        //return -1;
    }
    std::cout << "yaw_startpoint_deg :" << yaw_startpoint_deg << std::endl;

    yaw_startpoint = (yaw_startpoint_deg /180.0) * M_PI;
    //yaw_startpoint = ((124.0 - 157.3)/180.0) * M_PI;

    if (private_nh.getParam("pitch_startpoint_deg", pitch_startpoint_deg) == false)
    {
        std::cout << "pitch_startpoint_deg is not set." << std::endl;
        //return -1;
    }
    std::cout << "pitch_startpoint_deg : " << pitch_startpoint_deg << std::endl;

    pitch_startpoint = (pitch_startpoint_deg /180.0) * M_PI;


    if (private_nh.getParam("roll_startpoint_deg", roll_startpoint_deg) == false)
    {
        std::cout << "roll_startpoint_deg is not set." << std::endl;
        //return -1;
    }
    std::cout << "roll_startpoint_deg : " << roll_startpoint_deg << std::endl;

    roll_startpoint = (roll_startpoint_deg /180.0) * M_PI;



    // NDT parameters

    if (private_nh.getParam("ndt_grid_size", ndt_grid_size) == false)
    {
      std::cout << "ndt_grid_size is not set." << std::endl;
      //return -1;
    }
    std::cout << "ndt_grid_size: " << ndt_grid_size << std::endl;

    // downsampling grid size

    if (private_nh.getParam("voxel_leaf_size", voxel_leaf_size) == false)
    {
      std::cout << "voxel_leaf_size is not set." << std::endl;
      //return -1;
    }
    std::cout << "voxel_leaf_size: " << voxel_leaf_size << std::endl;

    if (private_nh.getParam("ndt_search_radius", ndt_search_radius) == false)
    {
      std::cout << "ndt_search_radius is not set." << std::endl;
      //return -1;
    }
    std::cout << "ndt_search_radius: " << ndt_search_radius << std::endl;



    // whether visualize or not

    if (private_nh.getParam("show_scan", show_scan) == false)
    {
      std::cout << "show_scan is not set." << std::endl;
      //return -1;
    }
    std::cout << "show_scan: " << show_scan << std::endl;

    if (private_nh.getParam("show_filtered_scan", show_filtered_scan) == false)
    {
      std::cout << "show_filtered_scan is not set." << std::endl;
      //return -1;
    }
    std::cout << "show_filtered_scan: " << show_filtered_scan << std::endl;

    if (private_nh.getParam("show_transformed_scan", show_transformed_scan) == false)
    {
      std::cout << "show_transformed_scan is not set." << std::endl;
      //return -1;
    }
    std::cout << "show_transformed_scan: " << show_transformed_scan << std::endl;

    if (private_nh.getParam("show_initial_scan", show_initial_scan) == false)
    {
      std::cout << "show_initial_scan is not set." << std::endl;
      //return -1;
    }
    std::cout << "show_initial_scan: " << show_initial_scan << std::endl;

    if (private_nh.getParam("show_map", show_map) == false)
    {
      std::cout << "show_map is not set." << std::endl;
      //return -1;
    }
    std::cout << "show_map: " << show_map << std::endl;

    if (private_nh.getParam("show_car_trajectory", show_car_trajectory) == false)
    {
      std::cout << "show_car_trajectory is not set." << std::endl;
      //return -1;
    }
    std::cout << "show_car_trajectory: " << show_car_trajectory << std::endl;


    // whether save output or not

    if (private_nh.getParam("save_transformed_scan", save_transformed_scan) == false)
    {
      std::cout << "save_transformed_scan is not set." << std::endl;
      //return -1;
    }
    std::cout << "save_transformed_scan: " << save_transformed_scan << std::endl;

    if (private_nh.getParam("save_pre_transformed_scan", save_pre_transformed_scan) == false)
    {
      std::cout << "save_pre_transformed_scan is not set." << std::endl;
      //return -1;
    }
    std::cout << "save_pre_transformed_scan: " << save_pre_transformed_scan << std::endl;

    if (private_nh.getParam("save_predicted_scan", save_predicted_scan) == false)
    {
      std::cout << "save_predicted_scan is not set." << std::endl;
      //return -1;
    }
    std::cout << "save_predicted_scan: " << save_predicted_scan << std::endl;

    if (private_nh.getParam("save_aligned_scan", save_aligned_scan) == false)
    {
      std::cout << "save_aligned_scan is not set." << std::endl;
      //return -1;
    }
    std::cout << "save_aligned_scan: " << save_aligned_scan << std::endl;

    if (private_nh.getParam("save_path", save_path) == false)
    {
      std::cout << "save_path is not set." << std::endl;
      //return -1;
    }
    std::cout << "save_path: " << save_path << std::endl;

    if (private_nh.getParam("save_transformed_dis_scan", save_transformed_dis_scan) == false)
    {
      std::cout << "save_transformed_dis_scan is not set." << std::endl;
      //return -1;
    }
    std::cout << "save_transformed_dis_scan: " << save_transformed_dis_scan << std::endl;



    // map parameters

    if (private_nh.getParam("map_file_path", map_file_path) == false)
    {
      std::cout << "map_file_path is not set." << std::endl;
      //return -1;
    }
    std::cout << "map_file_path: " << map_file_path << std::endl;

    if (private_nh.getParam("map_file_name", map_file_name) == false)
    {
      std::cout << "map_file_name is not set." << std::endl;
      //return -1;
    }
    std::cout << "map_file_name: " << map_file_name << std::endl;

    if (private_nh.getParam("map_load_mode", map_load_mode) == false)
    {
      std::cout << "map_load_mode is not set." << std::endl;
      //return -1;
    }
    std::cout << "map_load_mode: " << map_load_mode << std::endl;



    // do iterative distortion removal or not

    if (private_nh.getParam("do_iterative_distortion_removal", do_iterative_distortion_removal) == false)
    {
      std::cout << "do_iterative_distortion_removal is not set." << std::endl;
      //return -1;
    }
    std::cout << "do_iterative_distortion_removal: " << do_iterative_distortion_removal << std::endl;



    // do matching distortion removal or not

    if (private_nh.getParam("do_matching_distortion_removal", do_matching_distortion_removal) == false)
    {
      std::cout << "do_matching_distortion_removal is not set." << std::endl;
      //return -1;
    }
    std::cout << "do_matching_distortion_removal: " << do_matching_distortion_removal << std::endl;



    // other parameters

    if (private_nh.getParam("velodyne_yaw_offset", velodyne_yaw_offset) == false)
    {
      std::cout << "velodyne_yaw_offset is not set." << std::endl;
      //return -1;
    }
    std::cout << "velodyne_yaw_offset: " << velodyne_yaw_offset << std::endl;

    if (private_nh.getParam("use_prediction_from_file", use_prediction_from_file) == false)
    {
      std::cout << "use_prediction_from_file is not set." << std::endl;
      //return -1;
    }
    std::cout << "use_prediction_from_file: " << use_prediction_from_file << std::endl;

    if (private_nh.getParam("save_prediction_to_file", save_prediction_to_file) == false)
    {
      std::cout << "save_prediction_to_file is not set." << std::endl;
      //return -1;
    }
    std::cout << "save_prediction_to_file: " << save_prediction_to_file << std::endl;

    if (private_nh.getParam("predicted_pose_path", predicted_pose_path) == false)
    {
      std::cout << "predicted_pose_path is not set." << std::endl;
      //return -1;
    }
    std::cout << "predicted_pose_path: " << predicted_pose_path << std::endl;






    // make some string for file naming

    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << ndt_grid_size;
    std::string s = stream.str();

    std::stringstream stream2;
    stream2 << std::fixed << std::setprecision(1) << ndt_search_radius;
    std::string s2 = stream2.str();

    std::stringstream stream3;
    stream3 << std::fixed << std::setprecision(1) << voxel_leaf_size;
    std::string s3 = stream3.str();

    std::string name_tail = "ndtGrid(" + s + ")_searchRadius(" + s2 + ")_downSampling(" + s3 + ")";


    // Make necesssary folders

    if (true)
    {
        struct stat st = {0};

        std::string name = save_path;
        //save_path[save_path.size()] = ' ';

        if (stat(name.c_str(), &st) == -1)
        {
            mkdir(name.c_str(), 0700);
        }

        save_path = save_path + name_tail + "/";

        name = save_path;

        if (stat(name.c_str(), &st) == -1)
        {
            mkdir(name.c_str(), 0700);
        }
        else
        {
            ROS_ERROR("%s FOLDER ALREADY EXIST!!", save_path);
            return -1;
        }

        name = save_path + "predicted_scan";

        if (stat(name.c_str(), &st) == -1)
        {
            mkdir(name.c_str(), 0700);
        }

        name = save_path + "aligned_scan";

        if (stat(name.c_str(), &st) == -1)
        {
            mkdir(name.c_str(), 0700);
        }

        name = save_path + "transformed_scan";

        if (stat(name.c_str(), &st) == -1)
        {
            mkdir(name.c_str(), 0700);
        }

        name = save_path + "pre_transformed_scan";

        if (stat(name.c_str(), &st) == -1)
        {
            mkdir(name.c_str(), 0700);
        }

        name = save_path + "transformed_dis_scan";

        if (stat(name.c_str(), &st) == -1)
        {
            mkdir(name.c_str(), 0700);
        }



    }

    // make new files because later we add to the file

    FILE *pFile;

    std::string name = save_path + "position_log.csv";

    pFile = fopen (name.c_str(), "w");

    fprintf (pFile, "\
             (int)input->header.seq, \
             (float)current_pose.x,\
             (float)current_pose.y,\
             (float)current_pose.z, \
             (float)current_pose.roll,\
             (float)current_pose.pitch,\
             (float)current_pose.yaw, \
             \
             (int)ndt.hasConverged(),\
             (int)ndt.getFinalNumIteration(),\
             (float)ndt.getTransformationProbability(), \
             (float)matchingTime,\
             (int)scan.size(), \
             (int)filtered_scan.size(),\
             (float)initial_guess_error,\
             (float)initial_guess_yaw_error,\
             (float)vehicle_displacement,\
             (float)align_improvement_after_dis\
             \n");


    fclose (pFile);

    // make new file for predicted pose list

    if (save_prediction_to_file)
    {
        FILE *pFile;
        std::string name = save_path + "predicted_pose_list.csv";

        pFile = fopen(name.c_str(), "w");

        fprintf(pFile, "input->header.seq, temp_pose.x, temp_pose.y, temp_pose.z, \
                temp_pose.roll, temp_pose.pitch, temp_pose.yaw\n");

        fclose(pFile);
    }


    name = save_path + "x_points_histogram.csv";
    pFile = fopen (name.c_str(), "w");
    fclose (pFile);

    name = save_path + "y_points_histogram.csv";
    pFile = fopen (name.c_str(), "w");
    fclose (pFile);


    name = save_path + "layout_histogram_nonground.csv";
    pFile = fopen (name.c_str(), "w");
    fclose (pFile);


    name = save_path + "nonground_dirty_centroids_dimensions_values.csv";
    pFile = fopen (name.c_str(), "w");
    fclose (pFile);


    name = save_path + "yaw_and_NDT_dif_histogram.csv";
    pFile = fopen (name.c_str(), "w");
    fclose (pFile);


    if (!load_pointcloud_map<PointXYZIT>(map_file_path.c_str(), map_file_name.c_str(), map_load_mode, map_ptr))
    {
        std::cout << "error occured while loading map from following path :"<< std::endl;
        //std::cout << "map_file_path << std::endl;
    }
    else
    {
        map_loaded = true;
    }

    current_pose.x = x_startpoint;
    current_pose.y = y_startpoint;
    current_pose.z = z_startpoint;
    current_pose.roll = roll_startpoint;
    current_pose.pitch = pitch_startpoint;
    current_pose.yaw = yaw_startpoint;

    previous_pose = current_pose;

    offset.x = 0.0;
    offset.y = 0.0;
    offset.z = 0.0;
    offset.roll = 0.0;
    offset.pitch = 0.0;
    offset.yaw = 0.0;

    carPoseList.push_back(current_pose);

    // Setting NDT parameters to default values

    ndt.setMaximumIterations(ndt_iter);
    ndt.setResolution(ndt_grid_size);
    ndt.setSearchResolution(ndt_search_radius);
    ndt.setStepSize(ndt_step_size);
    ndt.setTransformationEpsilon(ndt_trans_eps);

    // LOAD MAP AS TARGET ###################################################################################
    ndt.setInputTarget(map_ptr);

    // LOAD PREDICTION FROM FILE TO POSE LIST################################################################

    if (use_prediction_from_file)
    {

        std::ifstream  data(predicted_pose_path);

        if (!data.is_open())
        {
            ROS_ERROR("\n\nError opening predicted pose listfor \n%s", predicted_pose_path);
            return -1;
        }


        std::string line;

        pose_entity pose_entity_temp;

        // first line is header so we need to skip one line
        std::getline(data,line);

        while(std::getline(data,line))
        {
            std::stringstream  lineStream(line);
            std::string        cell;

            // 1 --> seq
            std::getline(lineStream,cell,',');
            pose_entity_temp.seq = atof(cell.c_str());

            // 2 --> X
            std::getline(lineStream,cell,',');
            pose_entity_temp._pose.x = atof(cell.c_str());

            // 3 --> Y
            std::getline(lineStream,cell,',');
            pose_entity_temp._pose.y = atof(cell.c_str());

            // 4 --> Z
            std::getline(lineStream,cell,',');
            pose_entity_temp._pose.z = atof(cell.c_str());

            // 5 --> ROLL
            std::getline(lineStream,cell,',');
            pose_entity_temp._pose.roll = atof(cell.c_str());

            // 6 --> PITCH
            std::getline(lineStream,cell,',');
            pose_entity_temp._pose.pitch = atof(cell.c_str());

            // 7 --> YAW
            std::getline(lineStream,cell,',');
            pose_entity_temp._pose.yaw = atof(cell.c_str());

            predicted_pose_list.push_back(pose_entity_temp);

        }


    }


    /*
    pFileGlobalPositionCSV = fopen (strNameGlobalPositionCSV,"w");
    pFileRelativePositionCSV = fopen (strNameRelativePositionCSV,"w");
    pFileStateCSV = fopen (strNameStateCSV,"w");
    pFileLogCSV = fopen (strNameLogCSV,"w");
    //pFilePositionKML = fopen (strNamePositionKML,"w");
*/
/*
    fprintf (pFileGlobalPositionCSV, "#Absolute Positions : x,     y,    z\n");
    fprintf (pFileRelativePositionCSV, "#Relative Positions : x,     y,    z\n");
    fprintf (pFileStateCSV, "#Absolute Positions : x,    y,    z,     roll,    pith,    yaw    \n");
    fprintf (pFileLogCSV, "#logs : seq, relx, rely, relz, relroll, relpith, relyaw, converged, fitness score, probability, iteration number, reliability, exec time\n");

    std::cout << "Yaw initial   " << yaw_startpoint << std::endl ;
*/

    // Publish these topics

    map_pub = nh.advertise<sensor_msgs::PointCloud2>("/map", 1000);
    initialguess_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/initial_scan", 1000);
    transformed_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/transformed_scan", 1000);
    calibrated_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/calibrated_scan", 1000);
    scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/scan", 1000);
    predicted_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/predicted_scan", 1000);
    filtered_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/filtered_scan", 1000);
    aligned_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned_scan", 1000);
    transformed_dis_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/transformed_dis_scan", 1000);
    origin_dirty_centroids_pub = nh.advertise<sensor_msgs::PointCloud2>("/origin_dirty_centroids", 1000);
    dirty_centroids_pub = nh.advertise<sensor_msgs::PointCloud2>("/dirty_centroids", 1000);
    ground_dirty_centroids_pub = nh.advertise<sensor_msgs::PointCloud2>("/ground_dirty_centroids", 10);
    nonground_dirty_centroids_pub = nh.advertise<sensor_msgs::PointCloud2>("/nonground_dirty_centroids", 10);
    dirty_centroids_origin_pub = nh.advertise<sensor_msgs::PointCloud2>("dirty_centroids_origin", 10);
    map_in_velodyne_range_pub = nh.advertise<sensor_msgs::PointCloud2>("map_in_velodyne_range", 10);
    points_x_pub = nh.advertise<sensor_msgs::PointCloud2>("points_x", 10);
    points_y_pub = nh.advertise<sensor_msgs::PointCloud2>("points_y", 10);
    useful_oned_centroids_pub = nh.advertise<sensor_msgs::PointCloud2>("useful_oned_centroids", 10);;

    ndt_map_pub = nh.advertise<visualization_msgs::MarkerArray>("visualization_marker_array", 10000);
    carTrajectory_pub = nh.advertise<visualization_msgs::Marker>("CarTrajectory3D_line", 1000);


    // Subscribe to these topics

    ros::Subscriber scan_sub = nh.subscribe("velodyne_points", 5000, scan_callback);



    publish_pointCloud(*map_ptr, map_pub, "map");




    ros::spin();

    return 0;
}

