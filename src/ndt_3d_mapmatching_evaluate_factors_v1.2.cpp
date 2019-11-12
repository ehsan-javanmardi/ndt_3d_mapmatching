/*
Basic map matching with NDT class


 Ehsan Javanmardi


2018/7/10
    ndt_3d_mapmatching_evaluate_factors_v1.2
    calcualte factors based on the occupancy depth

2018/4/8
    ndt_3d_mapmatching_evaluate_factors_v1.1.1
    use new version of entropy.h
    dirty_centroid --> ndt_cell
    dirty_centroids --> dirty_ndt
    dirty_centroids_pcl --> dirty_centroids
    added dimension_ndt for visualization (now we have ndt_map with fixed color and dimension_ndt with 3 color)
    added vehicle model


2018/1/15
    ndt_3d_mapmatching_evaluate_factors_v1.0.8
    added first_scan to synch error and factors
    added generation_interval

2018/1/13
    ndt_3d_mapmatching_evaluate_factors_v1.0.7
    save folder ois based on parameters

2018/1/12
    ndt_3d_mapmatching_evaluate_factors_v1.0.6
    added occupancy degree

2018/1/11
    ndt_3d_mapmatching_evaluate_factors_v1.0.5
    added evaluate count
    added pfh_similarity
    added bhattacharyya distance similarity

2018/1/8
    ndt_3d_mapmatching_evaluate_factors_v1.0.4

2018/1/4
    ndt_3d_mapmatching_evaluate_factors_v1.0.3
    Show downsampled map on rviz
    Load map in main function
    added some new factors and vehicle pose to factors and changed the log names
    downsample the map and use the downsampled map


2018/1/3
    ndt_3d_mapmatching_evaluate_factors_v1.0.1
    Show dirty ndt in /dirty_ndt topic

2017/12/28
    ndt_3d_mapmatching_evaluate_factors_v1.0
    based on ndt_3d_mapmatching_v5.1.cpp

2017.12.21
    ndt_3D_mapmatching_v5.1
    Option: get groundtruth as initila_guess : (groundtruth_as_prediction)


 2017.12.17
    ndt_3D_mapmatching_v5.0
    based on ndt_3D_mapmatching_v4.0

 2017.07.14
 ndt_3D_mapmatching_v4.0

 TETS :
    code is not tested yet

 CHANGE LOG :
   Basic map matching

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
//#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/voxel_grid.h>
#include <math.h>
#include <boost/filesystem.hpp>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <ndt/visual_ndt.h>

#include "entropy.h"
#include "kmj_self_driving_common.h"
#include "self_driving_point_type.h"
#include <pcl/features/fpfh.h>
#include <pcl/features/pfh.h>

#include <nav_msgs/Odometry.h>

using namespace boost::filesystem;

#define PREDICT_POSE_THRESHOLD 0.5

#define Wa 0.4
#define Wb 0.3
#define Wc 0.3

pose initial_pose, predict_pose, previous_pose, ndt_pose;
pose current_pose, control_pose, localizer_pose, previous_gnss_pose, current_gnss_pose;
pose offset; // current_pos - previous_pose

// If the map is loaded, map_loaded will be true.
bool map_loaded = false;

NormalDistributionsTransform_Visual<pcl::PointXYZI, pcl::PointXYZI> ndt;

double ndt_search_res = 1.0;
double ndt_grid_size = 1.0;


// Leaf size of VoxelGrid filter.

double voxel_leaf_size = 1.0;

// publishers

ros::Publisher map_pub;
ros::Publisher ndt_map_pub;
ros::Publisher dimension_ndt_pub;
ros::Publisher dirty_centroids_pub;
ros::Publisher new_dirty_centroids_pub;
ros::Publisher transformed_scan_pub;
ros::Publisher initial_scan_pub;
ros::Publisher calibrated_scan_pub;
ros::Publisher scan_pub;
ros::Publisher predicted_scan_pub;
ros::Publisher filtered_scan_pub;
ros::Publisher carTrajectory_pub;
ros::Publisher aligned_scan_pub;
ros::Publisher transformed_dis_scan_pub;
ros::Publisher dirty_ndt_pub;
ros::Publisher odom_pub;
ros::Publisher map_velorange_pub;
ros::Publisher map_veloview_pub;
ros::Publisher vis_pub;

ros::Publisher ground_dirty_ndt_pub;
ros::Publisher nonground_dirty_ndt_pub;
ros::Publisher normal_histogram_cloud_pub;

// for test
ros::Publisher pfh_similarity_entropy_pub;
ros::Publisher long_cloud_pub;
ros::Publisher lat_cloud_pub;
ros::Publisher pfh_similarity_manhattan_pub;
ros::Publisher normal_cloud_pub;
ros::Publisher depth_cloud_pub;

ros::Publisher pfh_sim_ndpub;
ros::Publisher pfh_manh_ndpub;
ros::Publisher bhat_dis_ndpub;
ros::Publisher mahlnbs_dis_ndpub;




// show data on rviz

bool show_scan = true;
bool show_filtered_scan = true;
bool show_transformed_scan = true;
bool show_initial_scan = true;
bool show_map = true;
bool show_car_trajectory = true;
bool show_transformed_dis_scan = true;



// save scan data

bool save_transformed_scan = false;
bool save_predicted_scan = false;
bool save_aligned_scan = false;
bool save_transformed_dis_scan = false;

std::string map_file_path; // should be null otherwise it will have error
std::string map_file_name;
std::string save_path = "/home/ehsan/temp/results/map_matching/";
int map_load_mode = 0;
bool downsample_map = false;
bool map_ds_size = 0.1;

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

double x_startpoint = 0.0;
double y_startpoint = 0.0;
double z_startpoint = 0.0;
double yaw_startpoint =  0.0;//(-45/180.0) * M_PI ;
double roll_startpoint =  0.0;//(0/180.0) * M_PI ;
double pitch_startpoint = 0.0;//(-33/180.0) * M_PI  ;

bool groundtruth_as_prediction = false;
std::string ground_truth_filename="/home/ehsan/temp/input_data/ground_truth.csv";

struct pose_entity
{
    pose _pose;
    int seq;
    double stamp;
};

Eigen::Matrix4f tf_ground_truth;
pose ground_truth_pose;
pose previous_ground_truth_pose;
std::vector<pose_entity> ground_truth_list;

bool first_scan = true;

FILE * pFileLog;

pcl::PCDWriter writer;

pcl::PointCloud<PointXYZI>::Ptr map_ptr (new pcl::PointCloud<PointXYZI>);
pcl::PointCloud<PointXYZI>::Ptr filtered_map_ptr (new pcl::PointCloud<PointXYZI>);

// these tow variable is for log file only to show where the point cloud is saved
std::string savedMap ="";
std::string savedRoadMarkingWindow = "";

double lidar_range = 100.0;
std::string in_vel_topic = "/topic_preprocessor/vel_scan";

double vel_x= 0.0, vel_y= 0.0, vel_z= 0.0, vel_roll= 0.0, vel_pitch= 0.0, vel_yaw = 0.0;

double dis = 0.0;

double generation_interval = 1.0;

bool calculate_mahalanobis_factor = true;

// if use_velo_view is true then the velodyne scan used for factor evaluation instead of part of the map
bool use_velo_view = false;

// visualizer for the vehicle 3d
visualization_msgs::Marker car;

static void scan_callback(const sensor_msgs::PointCloud2::ConstPtr& input)
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

    // CONVERT MESSSAGE TO POINT CLOUD ######################################################################

    pcl::PointCloud<velodyne_pointcloud::PointXYZIR> scan_xyzir;
    pcl::fromROSMsg(*input, scan_xyzir);

    pcl::PointCloud<velodyne_pointcloud::PointXYZIR> calibrated_scan_xyzir;

    // CALIBRATE POINT CLOUD SUCH THAT THE SENSOR BECOME PERPENDICULAR TO GROUND SURFACE ####################
    // THIS STEP HELPS TO REMOVE GROUND SURFACE MUCH EASIER

    pose pose_lidar(vel_x, vel_y, vel_z, vel_roll/180.0*M_PI, vel_pitch/180.0*M_PI, vel_yaw/180.0*M_PI);
    static Eigen::Matrix4f tf_lidar;

    pose_to_tf(pose_lidar, tf_lidar);

    pcl::transformPointCloud(scan_xyzir, calibrated_scan_xyzir, tf_lidar);

    publish_pointCloud(calibrated_scan_xyzir, calibrated_scan_pub, "/velodyne");

    // DO NOT REMOVE GROUND #################################################################################

    pcl::PointCloud<PointXYZI> scan;

    for (pcl::PointCloud<velodyne_pointcloud::PointXYZIR>::const_iterator item = calibrated_scan_xyzir.begin(); \
         item != calibrated_scan_xyzir.end(); item++)
    {
        PointXYZI p;

        p.x = (double) item->x;
        p.y = (double) item->y;
        p.z = (double) item->z;
        p.intensity = (double) item->intensity;
        //p.ring = item->ring;

        if (getR(p) > 1.0 && getR(p) < lidar_range)
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
    else if(groundtruth_as_prediction)
    {
        // EXTRACT CURRENT PREDICTION FROM PREDICTED POSE LIST #################

        std::vector<pose_entity>::const_iterator item = ground_truth_list.begin();

        int scan_seq = input->header.seq;

        if (scan_seq < item->seq)
        {
            ROS_WARN("scan with seq %i skipped to align the predicted file and scans", scan_seq);
            return;
        }
        else
        {
            while (item != ground_truth_list.end() && item->seq < scan_seq)
            {
                item++;
            }

            if (item->seq != scan_seq)
            {
                ROS_ERROR("scan seq and groundtruth pose seq are not same. %i != %i", item->seq, scan_seq);
                // Do nothing
                return;
            }
            else
            {
                ground_truth_pose = item->_pose;
                pose_to_tf(ground_truth_pose, tf_ground_truth);

                std::cout << "Use groundtruth list for " << item->seq << std::endl;

                predict_pose = ground_truth_pose;
                pose_to_tf(predict_pose, tf_predict);
            }
        }

        // check the distance between two ground truth

        pose movement = ground_truth_pose - previous_ground_truth_pose;
        dis += sqrt(pow(movement.x, 2) + pow(movement.y, 2) + pow(movement.z, 2));

        previous_ground_truth_pose = ground_truth_pose;


        if (dis <= generation_interval && !first_scan)
        {
            std::cout << '\r' << "scan seq: " << scan_seq << " grt_seq : " << item->seq << ". moved less than " << dis << " so skipped !" << std::flush;
            return;
        }
        else
        {
            std::cout << "\033[1;42m Movement is more than " << generation_interval <<"m. Calculate error distribution for this scan!! \033[0m"<< std::endl;
            std::cout<< "movement is : " << dis << std::endl;
            dis = 0.0;
            first_scan = false;
        }

    }
    else
    {
        // local to global using tf_ltob x,y,z,yaw(heading)
        // calibrate
        // estimate current position using previous and offset
        // Guess the initial gross estimation of the transformation

        //offset.roll = 0.0;
        //offset.pitch = 0.0;
        predict_pose = previous_pose + offset;

        pose_to_tf(predict_pose, tf_predict);
        pose_to_tf(previous_pose, tf_previous); // I think this is not used so we can later remove this line
    }

    // SHOW PREDICTED SCAN ###################################################################

    if (show_scan)
    {
        pcl::PointCloud<PointXYZI>  predicted_scan;

        pcl::transformPointCloud(scan, predicted_scan, tf_predict);

        publish_pointCloud(predicted_scan, predicted_scan_pub, "map");

        // SHOW CALIBRATED INITIAL POSE IN GLOBAL COORDINATE

        if (show_initial_scan)
        {
            publish_pointCloud(predicted_scan, initial_scan_pub, "map");
            show_initial_scan = false;
        }
    }

    // SAVE PREDICTED CALIBRATED SCAN ##################################################

    if (save_predicted_scan)
    {
        pcl::PointCloud<PointXYZI> predicted_scan;

        pcl::transformPointCloud(scan, predicted_scan, tf_predict);

        predicted_scan.height = 1;
        predicted_scan.width = predicted_scan.size();
        predicted_scan.points.resize (predicted_scan.width * predicted_scan.height);

        std::string name = save_path + "predicted_scan_" + \
                           std::to_string((input->header.seq)) + ".pcd";

        writer.write(name, predicted_scan, false);
    }

    // DOWNSAMPLE SCAN USING VOXELGRID FILTER ###############################################################

    downsample_start =  std::chrono::system_clock::now();

    pcl::PointCloud<PointXYZI>::Ptr input_cloud_ptr(new pcl::PointCloud<PointXYZI>(scan));
    pcl::PointCloud<PointXYZI> filtered_scan;

    pcl::ApproximateVoxelGrid<PointXYZI> voxel_grid_filter;
    voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
    voxel_grid_filter.setInputCloud(input_cloud_ptr);
    voxel_grid_filter.filter(filtered_scan);

    downsample_end =  std::chrono::system_clock::now();

    downSampleTime = std::chrono::duration_cast<std::chrono::microseconds>\
                                        (downsample_end - downsample_start).count()/1000.0;

    // ALIGN POINT CLOUD TO THE MAP ##################################################################

    align_start = std::chrono::system_clock::now();

    //pcl::PointCloud<pcl::PointXYZI>::Ptr scan_ptr(new pcl::PointCloud<pcl::PointXYZI>(scan));

    pcl::PointCloud<PointXYZI>::Ptr filter_scan_ptr(new pcl::PointCloud<PointXYZI>(filtered_scan));

    ndt.setInputSource(filter_scan_ptr);

    pcl::PointCloud<PointXYZI> aligned_scan;

    ndt.align(aligned_scan, tf_predict);

    // get the global translation matrix

    Eigen::Matrix4f tf_align(Eigen::Matrix4f::Identity()); // base_link
    tf_align = ndt.getFinalTransformation(); // localizer

    tf_to_pose(tf_align, current_pose);

    int iteration = ndt.getFinalNumIteration();

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

    align_end = std::chrono::system_clock::now();

    alignTime = std::chrono::duration_cast<std::chrono::microseconds>\
                                        (align_end - align_start).count()/1000.0;

    matchingTime = alignTime + downSampleTime;

    // publish odom

    nav_msgs::Odometry odom;
    odom.header.stamp = current_scan_time;
    odom.header.frame_id = "map";

    //set the position
    odom.pose.pose.position.x = current_pose.x;
    odom.pose.pose.position.y = current_pose.y; // + rand() % 2 added some random noise to check filter
    odom.pose.pose.position.z = current_pose.z;

    //geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(current_pose.yaw);
    geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromRollPitchYaw(current_pose.roll,\
                                                                                  current_pose.pitch,\
                                                                                  current_pose.yaw);

    odom.pose.pose.orientation = odom_quat;

    //set the velocity
    odom.child_frame_id = "base_link";

    odom.twist.twist.linear.x = 0.0;
    odom.twist.twist.linear.y = 0.0;
    odom.twist.twist.linear.z = 0.0;
    odom.twist.twist.angular.x = 0.0;
    odom.twist.twist.angular.y = 0.0;
    odom.twist.twist.angular.z = 0.0;

    //publish the message
    odom_pub.publish(odom);


    // SHOW RESULTS ON THE SCREEN ##########################################################

    std::cout << "##############     sequence " << input->header.seq << "    ##############" << std::endl;
    std::cout << "X : " << current_pose.x << std::endl;
    std::cout << "Y : " << current_pose.y << std::endl;
    std::cout << "Y : " << current_pose.z << std::endl;
    std::cout << "yaw : " << current_pose.yaw << std::endl;

    //std::cout << "DownSampleTime : " << downSampleTime << std::endl;
    //std::cout << "AlignTime : " << alignTime << std::endl;
    std::cout << "MatchingTime : " << matchingTime << std::endl;
    std::cout << "Number of iteration : " << ndt.getFinalNumIteration() << std::endl;
    std::cout << "trans_probability : " << trans_probability << std::endl;
    std::cout << "Size of input points after downsampling : " << input_point_size << std::endl;

    //std::cout << "3D Displacement (current - previous) in meter " << displacement_3d << std::endl;
    //std::cout << "Error of initial guess (3D) in meter " << initial_guess_error << std::endl;
    //std::cout << "Yaw error of initial guess in degree " << initial_guess_yaw_error << std::endl;





    // SHOW DOWNSAMPLED SCAN ################################################################################

    if (show_filtered_scan)
    {
        pcl::PointCloud<PointXYZI> transformed_filtered_scan;

        pcl::transformPointCloud(filtered_scan, transformed_filtered_scan, tf_align);
        publish_pointCloud(transformed_filtered_scan, filtered_scan_pub, "map");
    }

    // SHOW ALIGNED SCAN ####################################################################################

    publish_pointCloud(aligned_scan, aligned_scan_pub, "map");

    // SAVE ALIGNED SCAN ######################################################

    if (save_aligned_scan)
    {
        aligned_scan.height = 1;
        aligned_scan.width = aligned_scan.size();
        aligned_scan.points.resize (aligned_scan.width * aligned_scan.height);

        std::string name = save_path + "aligned_scan_" + \
                           std::to_string((input->header.seq)) + ".pcd";

        writer.write(name, aligned_scan, false);

        if (filtered_scan.size() != aligned_scan.size())
        {
            PCL_ERROR("number of filtered_scan and aligned_scan are not same");
        }
    }

    pcl::PointCloud<PointXYZI> transformed_scan;

    pcl::transformPointCloud(scan, transformed_scan, tf_align);

    transformed_scan.height = 1;
    transformed_scan.width = transformed_scan.size();
    transformed_scan.points.resize (transformed_scan.width * transformed_scan.height);

    if (show_transformed_scan)
    {
        publish_pointCloud(transformed_scan, transformed_scan_pub, "map");
    }

    if (save_transformed_scan)
    {
        std::string name = save_path + "transformed_scan_" + \
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

        std::string name = save_path + "vehicle_pose.csv";

        FILE* pFile;
        pFile = fopen(name.c_str(), "a");

        fprintf (pFile, "%i,%f,%f,%f,%f,%f,%f,%f,%i,%f,%f,%i,%f,%f,%f,%i,%i,%f,%f\n",\
                 (int)input->header.seq, \
                 (float)current_scan_time.toSec(),\
                 (float)current_pose.x,\
                 (float)current_pose.y,\
                 (float)current_pose.z, \
                 (float)current_pose.roll,\
                 (float)current_pose.pitch,\
                 (float)current_pose.yaw, \
                 (int)ndt.hasConverged(),\
                 (float)ndt.getFitnessScore(), \
                 (float)ndt.getTransformationProbability(), \
                 (int)ndt.getFinalNumIteration(),\
                 (float)downSampleTime,\
                 (float)alignTime,\
                 (float)matchingTime,\
                 (int)scan.size(), \
                 (int)transformed_scan.size(),\
                 (float)initial_guess_error,\
                 (float)initial_guess_yaw_error);

        fclose(pFile);

    }

    // update vehicle visualizer pose

    car.pose.position.x = current_pose.x;
    car.pose.position.y = current_pose.y;
    car.pose.position.z = current_pose.z - 2.4;
    car.pose.orientation = odom_quat;

    vis_pub.publish(car);

    // BROADCAST VEHICLE TF SO THAT WE CAN SEE THE DRIVING VIEW IN RVIZ #####################################

    // currently does not work !!
    // maybe it works if we reset it

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion current_q;

    current_q.setRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);

    transform.setOrigin(tf::Vector3(current_pose.x, current_pose.y, current_pose.z));
    transform.setRotation(current_q);

    ros::Time current_scan_time;

    current_scan_time = input->header.stamp;

    br.sendTransform(tf::StampedTransform(transform, current_scan_time, "/map", "/base_link"));

    // get all points in velodyne range

    pcl::PointCloud<pcl::PointXYZI> map_velorange;
    bool is_map_velorange = false;

    if (true)
    {
        pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;

        //kdtree.setInputCloud (filtered_map_ptr);
        kdtree.setInputCloud (map_ptr);

        pcl::PointXYZI center_point;

        center_point.x = tf_ground_truth(0,3);
        center_point.y = tf_ground_truth(1,3);
        center_point.z = tf_ground_truth(2,3);

        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;

        if ( kdtree.radiusSearch (center_point, lidar_range + ndt_grid_size, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
        {
            pcl::PointXYZI p;

            for (int i=0; i< pointIdxRadiusSearch.size(); i++)
            {
                //p = (*filtered_map_ptr)[pointIdxRadiusSearch[i]];
                p = (*map_ptr)[pointIdxRadiusSearch[i]];
                map_velorange.push_back(p);
            }

            is_map_velorange = true;
        }
        else
        {
            PCL_WARN("getTargetPointsInNeiborhood No points found near tf_guess\n" );
        }

        publish_pointCloud(map_velorange, map_velorange_pub, "map");

        std::cout << "velorange points before ds :"<< map_velorange.size() << std::endl;
    }



    // get all points in velodyne view

    pcl::PointCloud<pcl::PointXYZI> map_veloview;
    bool is_map_veloview = false;

    if (false)
    {

        pcl::PointCloud<pcl::PointXYZI>::Ptr map_velorange_ptr(new pcl::PointCloud<pcl::PointXYZI> (map_velorange));

        pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;

        kdtree.setInputCloud (map_velorange_ptr);

        // veloview is very heavy becasue it has redundancy so currently we comment it

        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;

        pointIdxRadiusSearch.clear();

        for (int i=0; i< aligned_scan.size(); i++)
        {
            if ( kdtree.radiusSearch (aligned_scan[i], ndt_grid_size, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
            {
                pcl::PointXYZI p;

                for (int i=0; i< pointIdxRadiusSearch.size(); i++)
                {
                    p = map_velorange[pointIdxRadiusSearch[i]];
                    map_veloview.push_back(p);
                }
            }
        }

        // remove duplicate

        std::cout << "veloview points before ds :"<< map_veloview.size() << std::endl;

        pcl::PointCloud<PointXYZI>::Ptr map_veloview_ptr(new pcl::PointCloud<PointXYZI>(map_veloview));


        pcl::VoxelGrid<PointXYZI> duplicate_filter;
        duplicate_filter.setLeafSize(0.05, 0.05, 0.05);
        duplicate_filter.setInputCloud(map_veloview_ptr);
        duplicate_filter.filter(map_veloview);

        std::cout << "veloview points after ds :"<< map_veloview.size() << std::endl;

        publish_pointCloud(map_veloview, map_veloview_pub, "map");

    }

    // get dirty centroids

    std::vector<ndt_cell> dirty_ndt, dirty_ndt_view;
    std::vector <pcl::VoxelGridCovariance<pcl::PointXYZI>::Leaf> dirty_leafList, dirty_leafList_view;

    if (true)
    {

        pcl::VoxelGridCovariance<pcl::PointXYZI> target_cells;

        ndt.getCellsNew(target_cells);

        // D_shape << D1, D2, D3;

        /*getNearestDirtyCentroidsWithNormal(target_cells, filtered_scan, dirty_ndt,\
                                           dirty_ndt_Dimensions_values,\
                                           dirty_ndt_Dimensions,\
                                           dirty_leafList,\
                                           tf_align, ndt_search_res);*/

        // if use disrty centroids means we consider the velodyne view not the range

        if (use_velo_view)
        {

            //getNearestDirtyCentroids_unique_r(target_cells, filtered_scan, dirty_ndt,\
                                     dirty_leafList, tf_ground_truth, ndt_search_res + 1.5);

            getNearestDirtyCentroids(target_cells, filtered_scan, dirty_ndt,\
                                     dirty_leafList, tf_ground_truth, ndt_search_res);

            //getNearestDirtyCentroids_unique(target_cells, filtered_scan, dirty_ndt,\
                                            dirty_leafList, tf_ground_truth, ndt_search_res);
        }
        else
        {
            //getCentroidsInRange(target_cells, dirty_ndt,\
                                     dirty_leafList, tf_ground_truth, lidar_range + ndt_search_res, 30.0);

            //2018.07.10 changed to this
            getCentroidsInRange(target_cells, dirty_ndt,\
                                     dirty_leafList, tf_ground_truth, lidar_range, 30.0);

        }

        // just to count the number of dirty centroids in this case and compare to the range only

        getNearestDirtyCentroids_unique_r(target_cells, filtered_scan, dirty_ndt_view,\
                                 dirty_leafList_view, tf_ground_truth, ndt_search_res);



        /*
        //show dimension

        visualization_msgs::MarkerArray dirty_SphereList;
        setCovarianceListMarker<pcl::VoxelGridCovariance<pcl::PointXYZI>::Leaf>(dirty_leafList, dirty_SphereList, \
                                                      4.605 ,"map", 1000);

        */

        visualization_msgs::MarkerArray dirty_SphereList;
        Eigen::Vector4d RGBA(1.0, 0.0, 1.0, 0.5);
        setCovarianceListMarker<pcl::VoxelGridCovariance<pcl::PointXYZI>::Leaf>(dirty_leafList, dirty_SphereList, \
                                                      4.605 ,"map", RGBA, 1000);

        dirty_ndt_pub.publish(dirty_SphereList);
    }

    // get ground dirty centroids
    // get non ground dirty centroids
    // get dirty centroids point cloud

    std::vector<ndt_cell> ground_dirty_ndt;
    std::vector<ndt_cell> nonground_dirty_ndt;

    pcl::PointCloud<pcl::PointXYZI> dirty_centroids;
    dirty_centroids.clear();

    if (true)
    {
        for (std::vector<ndt_cell>::const_iterator iter = dirty_ndt.begin(); iter != dirty_ndt.end(); iter++)
        {
            pcl::PointXYZINormal p = iter->centroid;
            int shape = iter->shape_dimension;

            if ( shape == 2 && fabs(p.normal_z) >= fabs(p.normal_x) && fabs(p.normal_z) >= fabs(p.normal_y))
                ground_dirty_ndt.push_back(*iter);
            else
            {
                nonground_dirty_ndt.push_back(*iter);
            }

            pcl::PointXYZI point;
            point.x = iter->centroid.x;
            point.y = iter->centroid.y;
            point.z = iter->centroid.z;

            dirty_centroids.push_back(point);
        }
    }

    publish_pointCloud(dirty_centroids, dirty_centroids_pub, "map");
    //publish_pointCloud(nonground_dirty_ndt, nonground_dirty_ndt_pub, "map");

    FILE* pFile;

    std::string name = save_path + "factors.csv";

    pFile = fopen(name.c_str(), "a");

    // calculate occupancy ratio
    // currently occupancy ratio is calculated based on dirty centroids and this cannot be very accurate
    // we need to calculate it based on points or at least downsampled points
    // we assume map is already downsampled with 10cm

    double occupancy_ratio = 0.0;

    if (true)
    {
        double lidar_rpm = 20.0;
        double lidar_hr_res = 0.4; // in case of 20Hz
        double lidar_vr_res = 2.0;
        double lidar_layer_num = 16.0;

        int u_bin = (int)(360.0 / lidar_hr_res);
        int v_bin = (int)lidar_layer_num;
        int uv_space[v_bin][u_bin];

        // initialize the array
        for (int i=0; i< v_bin; i++)
            for (int j=0; j< u_bin; j++)
                uv_space[i][j] = 0;

        int occupied_count = 0;

        for (int i=0; i < map_velorange.size(); i++)
        {
            pcl::PointXYZI point = map_velorange[i];

            pcl::PointXYZ p;
            p.x = point.x - tf_ground_truth(0,3);
            p.y = point.y - tf_ground_truth(1,3);

            // update the layout histogram
            double hr_angle = getTheta(p);

            if (hr_angle > 360.0 || hr_angle < 0.0)
            {
                PCL_ERROR("Out of range hr_angle");
                std::cout << "out of range hr_angle " << hr_angle << std::endl;
                break;
            }

            if ( hr_angle == 360.0)
                hr_angle = 0.0;

            double dx = point.x - tf_ground_truth(0,3);
            double dy = point.y - tf_ground_truth(1,3);
            double h  = point.z - tf_ground_truth(2,3);

            double r = sqrt(pow(dx, 2) + pow(dy, 2));
            double vr_angle = atan(h/r) *  180.0 / M_PI; // -90 ~ 90

            vr_angle += 90.0; // 0 ~ 180

            if (vr_angle > 180.0 || vr_angle < 0.0)
            {
                PCL_ERROR("Out of range vr_angle");
                std::cout << "out of range vr_angle " << hr_angle << std::endl;
                break;
            }

            // we only consider -15 to +15 which is 75 to 105
            if (vr_angle >= 75.0 && vr_angle <= 105.0)
            {
                vr_angle -= 75.0; // 0 ~ 30
                ++uv_space[int(vr_angle / lidar_vr_res)][int(hr_angle / lidar_hr_res)];
            }
        }

        for (int i=0; i< v_bin; i++)
            for (int j=0; j< u_bin; j++)
            {
                if (uv_space[i][j] > 0)
                    ++occupied_count;
            }

        occupancy_ratio = (double)occupied_count / (double)(u_bin*v_bin);

        std::cout << "occupancy_ratio is " << occupancy_ratio << std::endl;
        std::cout << "occupied_count is " << occupied_count << std::endl;
        std::cout << "all cells count is " << u_bin*v_bin << std::endl;

        // show occupancy depth image

        pcl::PointCloud<pcl::PointXYZI> depth_cloud;

        for (int i=0; i< v_bin; i++)
            for (int j=0; j< u_bin; j++)
            {
                pcl::PointXYZI point;

                if (uv_space[i][j] > 0)
                {
                    point.x = j/10.0;
                    point.y = i;
                    //point.z = (double)i/5.0 - 3.0;
                    point.z = 0.0;
                    point.intensity = uv_space[i][j];
                }
                else
                {
                    point.x = j/10.0;
                    point.y = i;
                    //point.z = (double)i/5.0 - 3.0;
                    point.z = 0.0;
                    point.intensity = 0.0;
                }

                depth_cloud.push_back(point);
            }

        publish_pointCloud(depth_cloud, depth_cloud_pub, "velodyne");

    }


    fprintf (pFile, "%i,", (int)input->header.seq);
    fprintf (pFile, "%f,", (float)current_scan_time.toSec());
    fprintf (pFile, "%f,", (float)ground_truth_pose.x);
    fprintf (pFile, "%f,", (float)ground_truth_pose.y);
    fprintf (pFile, "%f,", (float)ground_truth_pose.z);

    fprintf(pFile, "%f,", occupancy_ratio);

    // extract the dirty_centroids that can be seen by the car

    if (true)
    {
        double lidar_rpm = 20.0;
        double lidar_hr_res = 5.0; // assumed as 5.0
        double lidar_vr_res = 2.0;
        double lidar_layer_num = 16.0;

        int u_bin = (int)(360.0 / lidar_hr_res);
        int v_bin = (int)lidar_layer_num;
        double uv_space[v_bin][u_bin]; //distance
        ndt_cell uv_ndt[v_bin][u_bin];

        // initialize the array
        for (int i=0; i< v_bin; i++)
            for (int j=0; j< u_bin; j++)
                uv_space[i][j] = 0;

        int occupied_count = 0;

        for (int i=0; i < dirty_ndt.size(); i++)
        {
            pcl::PointXYZI feature_center;
            feature_center.x = dirty_ndt[i].centroid.x;
            feature_center.y = dirty_ndt[i].centroid.y;
            feature_center.z = dirty_ndt[i].centroid.z;

            pcl::PointXYZ view_center;
            view_center.x = feature_center.x - tf_ground_truth(0,3);
            view_center.y = feature_center.y - tf_ground_truth(1,3);

            // update the layout histogram
            double hr_angle = getTheta(view_center);

            if (hr_angle > 360.0 || hr_angle < 0.0)
            {
                PCL_ERROR("Out of range hr_angle");
                std::cout << "out of range hr_angle " << hr_angle << std::endl;
                break;
            }

            if ( hr_angle == 360.0)
                hr_angle = 0.0;

            double dx = feature_center.x - tf_ground_truth(0,3);
            double dy = feature_center.y - tf_ground_truth(1,3);
            double h  = feature_center.z - tf_ground_truth(2,3);

            double r = sqrt(pow(dx, 2) + pow(dy, 2));
            double vr_angle = atan(h/r) *  180.0 / M_PI; // -90 ~ 90

            vr_angle += 90.0; // 0 ~ 180

            if (vr_angle > 180.0 || vr_angle < 0.0)
            {
                PCL_ERROR("Out of range vr_angle");
                std::cout << "out of range vr_angle " << hr_angle << std::endl;
                break;
            }

            // we only consider -15 to +15 which is 75 to 105
            if (vr_angle >= 75.0 && vr_angle <= 105.0)
            {
                vr_angle -= 75.0; // 0 ~ 30

                double current_value = uv_space[int(vr_angle / lidar_vr_res)][int(hr_angle / lidar_hr_res)];
                if (current_value !=0 && r > current_value)
                {
                    std::cout << " ";
                }
                else
                {
                    uv_space[int(vr_angle / lidar_vr_res)][int(hr_angle / lidar_hr_res)] = r;
                    uv_ndt[int(vr_angle / lidar_vr_res)][int(hr_angle / lidar_hr_res)] = dirty_ndt[i];
                }
            }
        }

        // remove the previous dirty_ndt

        dirty_ndt.clear();
        dirty_centroids.clear();

        for (int i=0; i< v_bin; i++)
            for (int j=0; j< u_bin; j++)
            {
                if (uv_space[i][j] > 0)
                {
                    dirty_ndt.push_back(uv_ndt[i][j]);

                    pcl::PointXYZI point;
                    point.x = uv_ndt[i][j].centroid.x;
                    point.y = uv_ndt[i][j].centroid.y;
                    point.z = uv_ndt[i][j].centroid.z;
                    point.intensity = 1.0;

                    dirty_centroids.push_back(point);
                }
            }

        std::cout << dirty_centroids.size() << "  " << dirty_ndt.size() << std::endl;

        publish_pointCloud(dirty_centroids, new_dirty_centroids_pub , "map");

    }

    // show new dirty centroids as point cloud






    // Calculate dimension shape for non ground dirty centroids

    pcl::PointCloud<pcl::PointXYZI> lat_cloud;
    pcl::PointCloud<pcl::PointXYZI> long_cloud;

    if (true)
    {
        int D1_count, D2_count, D3_count, useful_D1_count;
        double sum_D1, sum_D2, sum_D3;
        double lateral_weight, long_weight;
        double average_r = 0.0;

        D1_count = D2_count = D3_count = useful_D1_count = 0;
        sum_D1 = sum_D2 = sum_D3 = 0.0;
        lateral_weight = long_weight = 0.0;

        for (std::vector<ndt_cell>::const_iterator iter = dirty_ndt.begin(); iter != dirty_ndt.end(); iter++)
        {
            Eigen::Vector3d d_vector = iter->dimension_values;
            pcl::PointXYZINormal p = iter->centroid;
            int shape = iter->shape_dimension;

            sum_D1 += d_vector(0);
            sum_D2 += d_vector(1);
            sum_D3 += d_vector(2);

            //std::cout << shape << ",";

            if (shape == 1)
                ++D1_count;

            if (shape == 2)
                ++D2_count;

            if (shape == 3)
                ++D3_count;

            /*if (shape == 1)
                if (fabs(p.normal_z) < fabs(p.normal_x) && fabs(p.normal_z) < fabs(p.normal_y))
                    ++useful_D1_count;*/

            long_weight += iter->long_weight;
            lateral_weight += iter->leteral_weight;

            pcl::PointXYZI point;

            point.x = iter->centroid.x;
            point.y = iter->centroid.y;
            point.z = iter->centroid.z;
            point.intensity = lateral_weight;

            lat_cloud.push_back(point);

            point.intensity = long_weight;

            long_cloud.push_back(point);

            // calculate average distance of features

            average_r += sqrt(pow(p.x - current_pose.x, 2) + \
                              pow(p.y - current_pose.y, 2) + \
                              pow(p.z - current_pose.z, 2));
        }

        double D1_count_ratio = (double)D1_count / (double)dirty_ndt.size();
        double D2_count_ratio = (double)D2_count / (double)dirty_ndt.size();
        double D3_count_ratio = (double)D3_count / (double)dirty_ndt.size();

        sum_D1 = sum_D1 / (double)dirty_ndt.size();
        sum_D2 = sum_D2 / (double)dirty_ndt.size();
        sum_D3 = sum_D3 / (double)dirty_ndt.size();

        long_weight = long_weight / (double)dirty_ndt.size();
        lateral_weight = lateral_weight / (double)dirty_ndt.size();
        average_r = average_r / (double)dirty_ndt.size();

        publish_pointCloud(lat_cloud, lat_cloud_pub , "map");
        publish_pointCloud(long_cloud, long_cloud_pub, "map");

        std::cout << "d1 : " << D1_count << "  d2 : " << D2_count << "  d3 : " << D3_count << std::endl;

        fprintf (pFile, "%i,", (int)dirty_ndt.size());
        fprintf (pFile, "%i,", (int)dirty_ndt_view.size());

        fprintf (pFile, "%i,", (int)nonground_dirty_ndt.size());

        fprintf (pFile, "%i,", (int)D1_count);
        fprintf (pFile, "%i,", (int)D2_count);
        fprintf (pFile, "%i,", (int)D3_count);

        fprintf (pFile, "%f,", (float)D1_count_ratio);
        fprintf (pFile, "%f,", (float)D2_count_ratio);
        fprintf (pFile, "%f,", (float)D3_count_ratio);

        fprintf (pFile, "%f,", (float)(sum_D1));
        fprintf (pFile, "%f,", (float)(sum_D2));
        fprintf (pFile, "%f,", (float)(sum_D3));

        fprintf (pFile, "%f,", (float)(long_weight));
        fprintf (pFile, "%f,", (float)(lateral_weight));
        fprintf (pFile, "%f,", (float)(average_r));

        int nongr_D2_count = 0;

        for (std::vector<ndt_cell>::const_iterator iter = nonground_dirty_ndt.begin(); iter != nonground_dirty_ndt.end(); iter++)
        {
            int shape = iter->shape_dimension;

            if (shape == 2)
                ++nongr_D2_count;
        }

        fprintf (pFile, "%i,", (int)(nongr_D2_count));

    }

    // calculate DOP

    if (true)
    {
        double mean_x = 0.0, mean_y = 0.0, mean_z = 0.0;

        for (std::vector<ndt_cell>::const_iterator iter = dirty_ndt.begin(); iter != dirty_ndt.end(); iter++)
        {
            pcl::PointXYZINormal p = iter->centroid;
            double dx= 0.0, dy= 0.0, dz = 0.0;

            dx = p.x - current_pose.x;
            dy = p.y - current_pose.y;
            dz = p.z - current_pose.z;

            mean_x += dx ;
            mean_y += dy ;
            mean_z += dz ;
        }

        std::cout << "mean_x : " << mean_x << " mean_y : " << mean_y << " mean_z: " << mean_z << std::endl;

        mean_x = mean_x / (double)dirty_ndt.size();
        mean_y = mean_y / (double)dirty_ndt.size();
        mean_z = mean_z / (double)dirty_ndt.size();

        std::cout << "mean_x : " << mean_x << " mean_y : " << mean_y << " mean_z: " << mean_z << std::endl;

        double variance_x=0.0 , variance_y=0.0 ,variance_z =0.0;

        for (std::vector<ndt_cell>::const_iterator iter = dirty_ndt.begin(); iter != dirty_ndt.end(); iter++)
        {
            pcl::PointXYZINormal p = iter->centroid;

            double dx = 0.0, dy = 0.0, dz = 0.0;

            dx = p.x - current_pose.x;
            dy = p.y - current_pose.y;
            dz = p.z - current_pose.z;

            double r = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));

            variance_x += pow((dx - mean_x) / r, 2);
            variance_y += pow((dy - mean_y) / r, 2);
            variance_z += pow((dz - mean_z) / r, 2);
        }

        std::cout << "variance_x : " << variance_x << " variance_y : " << variance_y << " variance_z : " << variance_z << std::endl;

        variance_x = variance_x / (double)dirty_ndt.size();
        variance_y = variance_y / (double)dirty_ndt.size();
        variance_z = variance_z / (double)dirty_ndt.size();

        std::cout << "variance_x : " << variance_x << " variance_y : " << variance_y << " variance_z : " << variance_z << std::endl;

        double PDOP = (double)sqrt(variance_x + variance_y + variance_z);

        fprintf (pFile, "%f,", (float)(PDOP));

        std::cout << "PDOP is : " << PDOP << std::endl;
    }


    // make the weight for all 2D's
    // weight is based on :
    //   1. angle between viewpoint and the normal
    //   2. distance to center
    //   3. eigen values

    if (true)
    {
        double weight_with_eigen = 0.0;
        double weight = 0.0;

        int d2_count = 0;

        for (std::vector<ndt_cell>::const_iterator iter = nonground_dirty_ndt.begin(); \
             iter != nonground_dirty_ndt.end(); iter++)
        {
            ++d2_count;

            int shape = iter->shape_dimension;
            pcl::PointNormal centroid_normal ;

            centroid_normal.normal[0] = iter->centroid.normal[0];
            centroid_normal.normal[1] = iter->centroid.normal[1];
            centroid_normal.normal[2] = iter->centroid.normal[2];
            centroid_normal.curvature = iter->centroid.curvature;

            pcl::PointXYZINormal centroid = iter->centroid;

            // get the vector to the centroid of the shape and the center of the vehicle

            Eigen::Vector3d veh_centroid_vector(centroid.x - current_pose.x, centroid.y - current_pose.y, centroid.z - current_pose.z);

            if (shape == 2)
            {
                // calculate the angle between each shape and veh_centroid_vector

                double cos_theta = veh_centroid_vector(0) * centroid_normal.normal_x +\
                                   veh_centroid_vector(1) * centroid_normal.normal_y +\
                                   veh_centroid_vector(2) * centroid_normal.normal_z ;

                double r1 = sqrt(pow(veh_centroid_vector(0),2) + pow(veh_centroid_vector(1),2) + pow(veh_centroid_vector(2),2));
                double r2 = sqrt(pow(centroid_normal.normal_x,2) + pow(centroid_normal.normal_y,2) + pow(centroid_normal.normal_z,2));

                cos_theta = cos_theta / r1 * r2;

                weight = 1/r1 * cos_theta;
                weight_with_eigen = (1/r1) * cos_theta * (1/centroid_normal.curvature);

            }
        }

        double weight_av = weight / (double)d2_count;
        double weight_with_eigen_av = weight_with_eigen / (double)d2_count;

        fprintf (pFile, "%f,", (float)weight);
        fprintf (pFile, "%f,", (float)weight_with_eigen);

        fprintf (pFile, "%f,", (float)weight_av);
        fprintf (pFile, "%f,", (float)weight_with_eigen_av);

        std::cout << "weight_av is " << weight_av << std::endl;
    }


    // calculate entropy for the layout angle

    if (true)
    {
        // Make layout

        int num_bin = 36;
        int histogram[num_bin]={};

        double min_bin_val = 0.0;
        double max_bin_val = 360.0;

        if (makeLayoutAngleHistogram(dirty_ndt, histogram,\
                                num_bin, min_bin_val, max_bin_val, \
                                tf_ground_truth) != 0)
        {
            PCL_ERROR("ERROR in FUNCTION makeLayoutAngleHistogram\n");
        }

        double layoutangle_histogram = 0.0;

        calculateEntropyFromHistogram(histogram, num_bin, layoutangle_histogram);

        fprintf (pFile, "%f,", (float)layoutangle_histogram);

        std::cout << "layoutangle_histogram is " << layoutangle_histogram << std::endl;
    }



    // calculate entropy for the layout angle only 2D and 1D

    if (true)
    {

        // Make layout

        int num_bin = 36;
        int histogram[num_bin]={};

        double min_bin_val = 0.0;
        double max_bin_val = 360.0;

        // only use 2D and useful 1D ndt in layout entropy

        std::vector<ndt_cell> nonground_1D_2D_dirty_ndt;

        for (int i=0; i< dirty_ndt.size(); i++)
        {
            if (dirty_ndt[i].shape_dimension == 2 ||\
                dirty_ndt[i].shape_dimension == 1  )
            {
                nonground_1D_2D_dirty_ndt.push_back(dirty_ndt[i]);
            }

        }

        if (makeLayoutAngleHistogram(nonground_1D_2D_dirty_ndt, histogram,\
                                num_bin, min_bin_val, max_bin_val, \
                                tf_ground_truth) != 0)
        {
            PCL_ERROR("ERROR in FUNCTION makeLayoutAngleHistogram\n");
        }

        /*for (int i=0; i<num_bin; i++)
        {
            fprintf (pFile, "%i,", histogram[i]);
        }*/

        double layoutangle_histogram_1d2d= 0.0;

        calculateEntropyFromHistogram(histogram, num_bin, layoutangle_histogram_1d2d);

        fprintf (pFile, "%f,", (float)layoutangle_histogram_1d2d);

        std::cout << "layoutangle_histogram_1d2d is " << layoutangle_histogram_1d2d << std::endl;

    }





    // occupancy based on dirty centroids

    if (true)
    {
        /*double deg_resolution = (ndt_grid_size / lidar_range) * 180.0 / M_PI;// in radian

        std::cout << "test" << std::endl;

        double max_yaw = 360.0 / deg_resolution;
        int yaw_bin_size = (int)max_yaw + 1;

        double max_vr_angle = 180.0 / deg_resolution; // 45.0 deg is divided to deg_resolution. a little bit more than 30 of velodyne
        int vr_angle_bin_size = (int)max_vr_angle + 1;

        int depth_image[vr_angle_bin_size][yaw_bin_size];

        for (int i=0; i< vr_angle_bin_size; i++)
            for (int j=0; j< yaw_bin_size; j++)
                depth_image[i][j] = 0;

        for (int i=0; i < dirty_ndt.size(); i++)
        {
            pcl::PointXYZINormal point = dirty_ndt[i].centroid;

            pcl::PointXYZ p;
            p.x = point.x - tf_ground_truth(0,3);
            p.y = point.y - tf_ground_truth(1,3);

            // update the layout histogram
            double hr_angle = getTheta(p);

            if (hr_angle > 360 || hr_angle < 0)
                    std::cout << "out of range hr_angle " << hr_angle << std::endl;

            double dx = point.x - tf_ground_truth(0,3);
            double dy = point.y - tf_ground_truth(1,3);
            double h  = point.z - tf_ground_truth(2,3);

            double r = sqrt(pow(dx, 2) + pow(dy, 2));
            double vr_angle = atan(h/r) *  180.0 / M_PI; // -90 ~ 90
            vr_angle += 90.0; // 0 ~ 180

            if (vr_angle > 180.0 || vr_angle < 0.0)
                    std::cout << "out of range vr_angle " << hr_angle << std::endl;

            depth_image[int(vr_angle / deg_resolution)][int(hr_angle / deg_resolution)] = 1;
        }

        // calculate the occupancy ratio
        // just consider -20 to 20 of lidar --> 70 to 110 deg

        int total = 0;
        int occupied = 0;

        pcl::PointCloud<pcl::PointXYZI> occupancy_cloud;

        for (int i=0; i< vr_angle_bin_size; i++)
            for (int j=0; j< yaw_bin_size; j++)
            {
                if (i * deg_resolution < 110.0 && 70 < i * deg_resolution)
                {
                    pcl::PointXYZI point;

                    point.x = 0.0;
                    point.y = (double)j/10.0 - ((double)vr_angle_bin_size /20.0);
                    point.z = (double)i/5.0 - 3.0;
                    point.intensity = 100.0;

                    ++total;

                    if (depth_image[i][j] == 1)
                    {
                        ++occupied;
                        point.intensity = 200.0;
                    }

                    occupancy_cloud.push_back(point);
                }
            }

        double occupancy_ratio = (double)occupied / (double)total;

        fprintf (pFile, "%f,", (float)occupancy_ratio);

        pcl::transformPointCloud(occupancy_cloud, occupancy_cloud, tf_ground_truth);

        publish_pointCloud(occupancy_cloud, depth_cloud_pub, "map");
*/
    }



    // Calculate XY points entropy

    if (true)
    {
        /*

        pcl::PointCloud<pcl::PointXYZINormal> origin_dirty_ndt;

        transformPointCloudToOrigin(dirty_ndt, origin_dirty_ndt,\
                                    velodyne_yaw_offset, tf_ground_truth);

        publish_pointCloud(origin_dirty_ndt, dirty_ndt_origin_pub,"velodyne_origin");

        int num_bin = 200;
        int x_histogram[num_bin]={};
        int y_histogram[num_bin]={};
        double min_bin_val = -100.0;
        double max_bin_val = 100.0;

        makeXYCentroidsHistogram(origin_dirty_ndt, x_histogram, y_histogram,
                              num_bin, min_bin_val, max_bin_val);

        double x_points_feature_entropy = 0;

        calculateEntropyFromHistogram(x_histogram, num_bin, x_points_feature_entropy);

        fprintf (pFile, "%f,", x_points_feature_entropy);

        double y_points_feature_entropy = 0.0;

        calculateEntropyFromHistogram(y_histogram, num_bin, y_points_feature_entropy);

        fprintf (pFile, "%f,", y_points_feature_entropy);

        */
    }


    // calculate normal histogram for ground centroids

    int normal_histogram_90[91][91] = {0};

    if (true)
    {
        for (std::vector<ndt_cell>::const_iterator iter = dirty_ndt.begin(); \
             iter != dirty_ndt.end(); iter++)
        {
            pcl::PointXYZINormal p = iter->centroid;

            double x = p.normal_x;
            double y = p.normal_y;
            double z = p.normal_z;

            double r = sqrt(pow(x,2) + pow(y,2));

            double theta, vr_angle;

            if (x!=0)
                theta = atan (y/x) * 180.0 / M_PI + 90.0;
            else
                theta = 180.0;

            //std::cout << theta << std::endl;
            int theta_index = int (floor(theta / 2.0)); // 2 is histogram interval

            if (r!=0)
                vr_angle = atan (z/r) * 180.0 / M_PI + 90.0;
            else
                vr_angle = 180.0;

            int vr_angle_index = int (floor(vr_angle / 2.0)); // 2 is histogram interval

            //std::cout <<vr_angle_index << " , " << theta_index << std::endl;

            ++normal_histogram_90[vr_angle_index][theta_index]; // -pi/2 to pi/2
        }
    }

    // calculate entropy of normal histogram

    double ndt_entropy_90 = 0.0;

    if (true)
    {

        int sum = 0;

        for (int i=0; i <= 90; i++)
            for (int j=0; j <= 90; j++)
                sum += normal_histogram_90[i][j];

        for (int i=0; i <= 90; i++)
            for (int j=0; j <= 90; j++)
            {
                if (normal_histogram_90[i][j] != 0)
                {

                    double px = (double)normal_histogram_90[i][j] / (double)sum;

                    double log_px = (double)(log2(px));

                    ndt_entropy_90 += (double)(-1.0 * log_px * px);
                }
            }

        std::cout  << "ndt entropy (90 bin) for grid size " << ndt_grid_size  << " is : " << ndt_entropy_90 << std::endl;

        fprintf (pFile, "%f,", (float)ndt_entropy_90);
    }

    // calculate normal histogram for ground centroids with 16 bins

    int normal_histogram_16[16][16] = {0};

    if (true)
    {
        pcl::PointCloud<pcl::PointXYZINormal> normal_cloud;

        for (std::vector<ndt_cell>::const_iterator iter = dirty_ndt.begin(); \
             iter != dirty_ndt.end(); iter++)
        {
            pcl::PointXYZINormal p = iter->centroid;

            double x = p.normal_x;
            double y = p.normal_y;
            double z = p.normal_z;

            double r = sqrt(pow(x,2) + pow(y,2));

            double theta, vr_angle;

            if (x!=0)
                theta = atan (y/x) * 180.0 / M_PI + 90.0;
            else
                theta = 180.0;

            //std::cout << theta << std::endl;
            int theta_index = int (floor(theta / 11.25)); // 2 is histogram interval
            if (theta_index == 16)
                theta_index = 15; // ensure that we have only 16 bins

            if (r!=0)
                vr_angle = atan (z/r) * 180.0 / M_PI + 90.0;
            else
                vr_angle = 180.0;

            int vr_angle_index = int (floor(vr_angle / 11.25)); // 2 is histogram interval
            if (vr_angle_index == 16)
                vr_angle_index = 15; // ensure that we have only 16 bins

            //std::cout <<vr_angle_index << " , " << theta_index << std::endl;

            ++normal_histogram_16[vr_angle_index][theta_index]; // -pi/2 to pi/2

            normal_cloud.push_back(p);
        }

        publish_pointCloud(normal_cloud, normal_cloud_pub, "map");

    }

    // calculate entropy of normal histogram

    double ndt_entropy_16 = 0.0;

    if (true)
    {

        int sum = 0;

        for (int i=0; i< 16; i++)
            for (int j=0; j<16; j++)
                sum += normal_histogram_16[i][j];

        for (int i=0; i< 16; i++)
            for (int j=0; j<16; j++)
            {
                if (normal_histogram_16[i][j] != 0)
                {

                    double px = (double)normal_histogram_16[i][j] / (double)sum;

                    double log_px = (double)(log2(px));

                    ndt_entropy_16 += (double)(-1.0 * log_px * px);
                }
            }

        std::cout  << "ndt entropy (16 bin) for grid size " << ndt_grid_size  << " is : " << ndt_entropy_16 << std::endl;

        fprintf (pFile, "%f,", (float)ndt_entropy_16);
    }


    // calculate normal histogram for ground centroids with 16 bins only for 2D

    int normal_histogram2d_16[16][16] = {0};

    if (true)
    {
        for (std::vector<ndt_cell>::const_iterator iter = dirty_ndt.begin(); \
             iter != dirty_ndt.end(); iter++)
        {
            // only 2D

            if (iter->shape_dimension != 2)
                continue;

            pcl::PointXYZINormal p = iter->centroid;

            double x = p.normal_x;
            double y = p.normal_y;
            double z = p.normal_z;

            double r = sqrt(pow(x,2) + pow(y,2));

            double theta, vr_angle;

            if (x!=0)
                theta = atan (y/x) * 180.0 / M_PI + 90.0;
            else
                theta = 180.0;

            //std::cout << theta << std::endl;
            int theta_index = int (floor(theta / 11.25)); // 2 is histogram interval
            if (theta_index == 16)
                theta_index = 15; // ensure that we have only 16 bins

            if (r!=0)
                vr_angle = atan (z/r) * 180.0 / M_PI + 90.0;
            else
                vr_angle = 180.0;

            int vr_angle_index = int (floor(vr_angle / 11.25)); // 2 is histogram interval
            if (vr_angle_index == 16)
                vr_angle_index = 15; // ensure that we have only 16 bins

            //std::cout <<vr_angle_index << " , " << theta_index << std::endl;

            ++normal_histogram2d_16[vr_angle_index][theta_index]; // -pi/2 to pi/2

        }
    }



    // calculate entropy of normal histogram

    double normal_entropy_2d_16 = 0.0;

    if (true)
    {

        int sum = 0;

        for (int i=0; i< 16; i++)
            for (int j=0; j<16; j++)
                sum += normal_histogram2d_16[i][j];

        for (int i=0; i< 16; i++)
            for (int j=0; j<16; j++)
            {
                if (normal_histogram2d_16[i][j] != 0)
                {

                    double px = (double)normal_histogram2d_16[i][j] / (double)sum;

                    double log_px = (double)(log2(px));

                    normal_entropy_2d_16 += (double)(-1.0 * log_px * px);
                }
            }

        std::cout  << "ndt entropy (16 bin) for grid size " << ndt_grid_size  << " is : " << normal_entropy_2d_16 << std::endl;

        fprintf (pFile, "%f,", (float)normal_entropy_2d_16);
    }



    // calculate normal histogram for ground centroids with 8 bins

    int normal_hitogram_8[8][8] = {0};

    if (true)
    {
        for (std::vector<ndt_cell>::const_iterator iter = dirty_ndt.begin(); \
             iter != dirty_ndt.end(); iter++)
        {
            pcl::PointXYZINormal p = iter->centroid;

            double x = p.normal_x;
            double y = p.normal_y;
            double z = p.normal_z;

            double r = sqrt(pow(x,2) + pow(y,2));

            double theta, vr_angle;

            if (x!=0)
                theta = atan (y/x) * 180.0 / M_PI + 90.0;
            else
                theta = 180.0;

            //std::cout << theta << std::endl;
            int theta_index = int (floor(theta / 11.25)); // 2 is histogram interval
            if (theta_index == 8)
                theta_index = 7; // ensure that we have only 8 bins

            if (r!=0)
                vr_angle = atan (z/r) * 180.0 / M_PI + 90.0;
            else
                vr_angle = 180.0;

            int vr_angle_index = int (floor(vr_angle / 11.25)); // 2 is histogram interval
            if (vr_angle_index == 8)
                vr_angle_index = 7; // ensure that we have only 8 bins

            //std::cout <<vr_angle_index << " , " << theta_index << std::endl;

            ++normal_hitogram_8[vr_angle_index][theta_index]; // -pi/2 to pi/2

        }
    }



    // calculate entropy of normal histogram

    double normal_entropy_8 = 0.0;

    if (true)
    {

        int sum = 0;

        for (int i=0; i< 8; i++)
            for (int j=0; j<8; j++)
                sum += normal_hitogram_8[i][j];

        for (int i=0; i< 8; i++)
            for (int j=0; j<8; j++)
            {
                if (normal_hitogram_8[i][j] != 0)
                {

                    double px = (double)normal_hitogram_8[i][j] / (double)sum;

                    double log_px = (double)(log2(px));

                    normal_entropy_8 += (double)(-1.0 * log_px * px);
                }
            }

        std::cout  << "ndt entropy (8 bin) for grid size " << ndt_grid_size  << " is : " << normal_entropy_8 << std::endl;

        fprintf (pFile, "%f,", (float)normal_entropy_8);
    }


    // calculate PFH for all centroids

    pcl::PointCloud<pcl::PFHSignature125>::Ptr pfhs (new pcl::PointCloud<pcl::PFHSignature125> ());
    pcl::PointCloud<pcl::PointXYZI> centroid_cloud;
    pcl::PointCloud<pcl::Normal> normal_cloud;

    if (true)
    {
        for (std::vector<ndt_cell>::const_iterator iter = dirty_ndt.begin(); \
             iter != dirty_ndt.end(); iter++)
        {
            // get centroids cloud and corresponding normals

            pcl::PointXYZI point;

            pcl::Normal normal;

            point.x = iter->centroid.x;
            point.y = iter->centroid.y;
            point.z = iter->centroid.z;

            normal.normal_x = iter->centroid.normal_x;
            normal.normal_y = iter->centroid.normal_y;
            normal.normal_z = iter->centroid.normal_z;

            centroid_cloud.push_back(point);
            normal_cloud.push_back(normal);
        }

        // calculate pfh for centroids

        pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;

        typename pcl::PointCloud<pcl::PointXYZI>::Ptr centroid_cloud_ptr (new pcl::PointCloud<pcl::PointXYZI> (centroid_cloud)) ;
        typename pcl::PointCloud<pcl::Normal>::Ptr normal_cloud_ptr (new pcl::PointCloud<pcl::Normal> (normal_cloud)) ;

        // Create the PFH estimation class, and pass the input dataset+normals to it
        pcl::PFHEstimation<pcl::PointXYZI, pcl::Normal, pcl::PFHSignature125> pfh;
        pfh.setInputCloud (centroid_cloud_ptr);
        pfh.setInputNormals (normal_cloud_ptr);

        // alternatively, if cloud is of tpe PointNormal, do pfh.setInputNormals (cloud);

        // Create an empty kdtree representation, and pass it to the PFH estimation object.
        // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
        pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI> ());
        //pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr tree (new pcl::KdTreeFLANN<pcl::PointXYZI> ()); -- older call for PCL 1.5-
        pfh.setSearchMethod (tree);

        // Use all neighbors in a sphere of radius 5cm
        // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
        pfh.setRadiusSearch (ndt_grid_size * 2.0);

        // Compute the features
        pfh.compute (*pfhs);

        // pfhs->points.size () should have the same size as the input cloud->points.size ()*
        // (int)(*pfhs)[i].histogram[j] are the percent not the count !!

        if (pfhs->points.size() != dirty_ndt.size())
            PCL_ERROR("Point size are not after pfh ca,culation");

    }

    // calculate similarity according to pfh of the points
    // first calculate entropy for each point
    // average of the entropy show the similariy
    // high entropy --> low similarity


    if (true)
    {
        double pfh_similarity_entropy = 0.0;

        pcl::PointCloud<pcl::PointXYZI> pfh_similarity_entropy_cloud;

        std::vector <double> entropy_list;

        for (int i=0; i< centroid_cloud.size(); i++)
        {
            double entropy = 0.0;

            for (int j = 0; j < 125; j++)
            {
                // calculate the entropy for each of them

                double px = (double)((*pfhs)[i].histogram[j]);
                px = px /100.0;

                if (px != 0.0)
                {
                    double log_px = (double)(log2(px));
                    entropy += (double)(-1.0 * log_px * px);
                }
            }

            pfh_similarity_entropy += entropy;

            pcl::PointXYZI point;
            point.x = centroid_cloud[i].x;
            point.y = centroid_cloud[i].y;
            point.z = centroid_cloud[i].z;
            point.intensity = entropy;

            pfh_similarity_entropy_cloud.push_back(point);

            entropy_list.push_back(entropy);
        }

        pfh_similarity_entropy = pfh_similarity_entropy / (double)centroid_cloud.size();

        fprintf (pFile, "%f,", (float)pfh_similarity_entropy);

        std::cout << "pfh_similarity_entropy is : " << pfh_similarity_entropy << std::endl;

        publish_pointCloud(pfh_similarity_entropy_cloud, pfh_similarity_entropy_pub, "map");

        // standardize entropy to get opacity value for visualization

        double min_bhat_dis = 10000.0;
        double max_bhat_dis = -10000.0;

        for (int i=0; i< entropy_list.size(); i++)
        {
            if (entropy_list[i] < min_bhat_dis)
                min_bhat_dis = entropy_list[i];

            if (entropy_list[i] > max_bhat_dis)
                max_bhat_dis = entropy_list[i];
        }

        std::vector<double> pfh_sim_H_list;
        double range = max_bhat_dis - min_bhat_dis ;

        for (int i=0; i< entropy_list.size(); i++)
        {
            double temp = entropy_list[i] - min_bhat_dis ;
            temp = temp / range ;

            pfh_sim_H_list.push_back(temp);
        }

        Eigen::Vector4d RGBA(0.0, 0.0, 0.0, 0.0);

        visualization_msgs::MarkerArray dirty_SphereList;

        setCovarianceListMarker_withOpacity<pcl::VoxelGridCovariance<pcl::PointXYZI>::Leaf>\
                (dirty_leafList, dirty_SphereList, 4.605 , RGBA, pfh_sim_H_list, "map", 1000);


        pfh_sim_ndpub.publish(dirty_SphereList);

    }



    // calcualte manhattan and battacharyya distance of the pfh

    if (true)
    {

        std::vector<double> pfh_sim_mnht_list;
        std::vector<double> bhat_dis_list;


        double pfh_similarity_manhattan = 0.0;
        double bhatt_d_similarity = 0.0;


        double manhattan_total = 0.0;
        double bhatt_d_total = 0.0; // total bhatacharyya  distance

        for (int i=0; i< dirty_ndt.size(); i++)
        {
            pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;

            typename pcl::PointCloud<pcl::PointXYZI>::Ptr centroid_cloud_ptr (new pcl::PointCloud<pcl::PointXYZI> (centroid_cloud)) ;

            kdtree.setInputCloud (centroid_cloud_ptr);

            // Neighbors within radius search

            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;

            pcl::PointXYZI search_point;
            search_point.x = dirty_ndt[i].centroid.x;
            search_point.y = dirty_ndt[i].centroid.y;
            search_point.z = dirty_ndt[i].centroid.z;

            double manhattan_dis = 0.0; // manhattan distance
            double bhatt_d = 0.0; // bhatacharyya distance

            if ( kdtree.radiusSearch (search_point, ndt_grid_size * 1.3, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
            {
               for (size_t j = 0; j < pointIdxRadiusSearch.size (); ++j)
               {
                   // calculate manhattan ditance of the histograms

                   for (int k=0; k < 125; k++)
                   {
                        manhattan_dis += fabs((*pfhs)[i].histogram[k] - (*pfhs)[pointIdxRadiusSearch[j]].histogram[k]);
                   }

                   // calculate the bhattacharyya distance between two distribution with same mean
                   pcl::VoxelGridCovariance<pcl::PointXYZI>::Leaf cell_1 = dirty_ndt[i].cell;
                   pcl::VoxelGridCovariance<pcl::PointXYZI>::Leaf cell_2 = dirty_ndt[pointIdxRadiusSearch[j]].cell;

                   Eigen::Matrix3d cell_1_cov = cell_1.getCov();
                   Eigen::Matrix3d cell_2_cov = cell_2.getCov();

                   bhatt_d += bhattacharyya(Eigen::Vector3d(0.0, 0.0, 0.0), cell_1_cov, cell_2_cov);
               }

               //std::cout << pointIdxRadiusSearch.size () << ",";
               manhattan_dis /= (double)pointIdxRadiusSearch.size ();
               bhatt_d /= (double)pointIdxRadiusSearch.size ();

            }

            pcl::PointXYZI point;
            point.x = dirty_ndt[i].centroid.x;
            point.y = dirty_ndt[i].centroid.y;
            point.z = dirty_ndt[i].centroid.z;
            point.intensity = manhattan_dis;

            pfh_sim_mnht_list.push_back(manhattan_dis);
            bhat_dis_list.push_back(bhatt_d);

            manhattan_total += manhattan_dis;
            bhatt_d_total += bhatt_d;

            //similarity += 1.0 / manhattan_dis;
        }

        pfh_similarity_manhattan = manhattan_total / dirty_ndt.size();
        bhatt_d_similarity = bhatt_d_total / dirty_ndt.size();


        if (pfh_sim_mnht_list.size() != dirty_ndt.size())
            PCL_ERROR("Point size are not after pfh ca,culation");

        std::cout << "pfh_similarity based on manhattan is : " << pfh_similarity_manhattan << std::endl;
        std::cout << "bhattachariyya distance based similarity is  : " << bhatt_d_similarity << std::endl;


        fprintf (pFile, "%f,", (float)pfh_similarity_manhattan);
        fprintf (pFile, "%f,", (float)bhatt_d_similarity);



        // show manhattan

        if (true)
        {
            Eigen::Vector4d RGBA(0.0, 0.0, 0.0, 0.0);

            visualization_msgs::MarkerArray dirty_SphereList;

            setCovarianceListMarker_withOpacity<pcl::VoxelGridCovariance<pcl::PointXYZI>::Leaf>\
                    (dirty_leafList, dirty_SphereList, 4.605 , RGBA, pfh_sim_mnht_list, "map", 1000);


            pfh_manh_ndpub.publish(dirty_SphereList);

        }

        if (true)
        {
            Eigen::Vector4d RGBA(0.0, 0.0, 0.0, 0.0);

            visualization_msgs::MarkerArray dirty_SphereList;

            setCovarianceListMarker_withOpacity<pcl::VoxelGridCovariance<pcl::PointXYZI>::Leaf>\
                    (dirty_leafList, dirty_SphereList, 4.605 , RGBA, bhat_dis_list, "map", 1000);


            bhat_dis_ndpub.publish(dirty_SphereList);

        }

    }

    std::cout << "test" << std::endl;

    // calculate occupancy degree

    if (false)
    {
        /*pcl::PointCloud<pcl::PointXYZI> dirty_centroids;

        for (int i=0; i < dirty_ndt.size(); i++)
        {
            pcl::PointCloud<pcl::PointXYZI> cloud;
            getPCLFromLeaf(dirty_ndt[i].cell, cloud, ndt_grid_size * 100.0, ndt_grid_size);
            dirty_centroids += cloud;
        }

        publish_pointCloud(dirty_centroids, pfh_similarity_manhattan_pub, "map");*/

        // make depth image

    }







    // calculate the generalization degree (environment change) by mahlanobis distance

    std::cout << "test1" << std::endl;

    if (calculate_mahalanobis_factor)
    {
        // downsample velodyne_range !!

        bool do_ds_velorange = false;

        pcl::PointCloud<PointXYZI> filtered_map_velorange = map_velorange;

        if (do_ds_velorange)
        {
            filtered_map_velorange.clear();

            pcl::PointCloud<PointXYZI>::Ptr map_velorange_ptr(new pcl::PointCloud<PointXYZI>(map_velorange));
            //pcl::VoxelGrid<PointXYZI> voxel_grid_filter;
            pcl::ApproximateVoxelGrid<PointXYZI> voxel_grid_filter;
            voxel_grid_filter.setLeafSize(0.2, 0.2, 0.2);
            voxel_grid_filter.setInputCloud(map_velorange_ptr);
            voxel_grid_filter.filter(filtered_map_velorange);

            std::cout << "map_velorange after downsample is : " << filtered_map_velorange.size() << std::endl;
        }

        // calculate the mahalanobis distance of each point with the dirty centroids
        // for each point calcualte the mahalanobis to the nearest dirty centroids

        pcl::PointCloud<pcl::PointXYZI>::Ptr dirty_centroids_ptr\
                (new pcl::PointCloud<pcl::PointXYZI> (dirty_centroids));
        pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
        kdtree.setInputCloud (dirty_centroids_ptr);

        double mahalanobis_dis = 0.0;
        double euclidean_dis = 0.0;

        // list of mahalanobis value for each centroid
        std::vector<double> mhln_per_ndt;

        // list of number of points used to calculate mahalanobis value for each centroid
        std::vector<int> npoint_per_ndt;

        for (int i = 0; i < dirty_centroids.size(); i++)
        {
            mhln_per_ndt.push_back(0.0);
            npoint_per_ndt.push_back(0);
        }

        // for each point in the map calculate mahalanobis distance

        int euclidean_dis_count = 0;
        int mahalanobis_dis_count = 0;

        for (typename pcl::PointCloud<pcl::PointXYZI>::const_iterator item = filtered_map_velorange.begin();\
             item!=filtered_map_velorange.end(); item++ )
        {

            // get point index

            int i = int (item->x / ndt_grid_size);
            int j = int (item->y / ndt_grid_size);
            int k = int (item->z / ndt_grid_size);

            // get the cneter of grid in which the point belong
            // we need to get the grid center and from grid center find corresponding centorid for this point
            // because in the abstraction phase the ND is made by the points inside the grid

            PointXYZI center;

            if (item->x >= 0)
                center.x = (double)i * ndt_grid_size + (ndt_grid_size / 2.0);
            else
                center.x = (double)i * ndt_grid_size - (ndt_grid_size / 2.0);


            if (item->y >= 0)
                center.y = (double)j * ndt_grid_size + (ndt_grid_size / 2.0);
            else
                center.y = (double)j * ndt_grid_size - (ndt_grid_size / 2.0);


            if (item->z >= 0)
                center.z = (double)k * ndt_grid_size + (ndt_grid_size / 2.0);
            else
                center.z = (double)k * ndt_grid_size - (ndt_grid_size / 2.0);

            // find the nearest centroid pcl

            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;

            kdtree.nearestKSearch (center, 1, pointIdxRadiusSearch, pointRadiusSquaredDistance);
            //kdtree.radiusSearch (center, ndt_grid_size*2.0, pointIdxRadiusSearch, pointRadiusSquaredDistance);

            if ( pointIdxRadiusSearch.size() == 0)
            {
                PCL_ERROR("point is zero!!!!!");
                continue;
            }

            // in the case radius search used, find the nearest centroid

            double min_dis = 1000.0;

            // index of centroid_pcl which we updated
            int _it = 0;

            for (int m=0; m< pointIdxRadiusSearch.size(); m++)
            {
                if (pointRadiusSquaredDistance[m] < min_dis)
                {
                    min_dis  = pointRadiusSquaredDistance[m];
                    _it = pointIdxRadiusSearch[m];
                }
            }

            PointXYZINormal mean = dirty_ndt[_it].centroid;

            pcl::VoxelGridCovariance<pcl::PointXYZI>::Leaf cell = dirty_ndt[_it].cell;

            Eigen::Vector3d d_mean;
            d_mean << ((double)(item->x - mean.x)) , ((double)(item->y - mean.y)), ((double)(item->z - mean.z));

            Eigen::Matrix3d inv_cov = cell.getInverseCov();

            //calculate mahalanobis of leaf and the point!!

            //Eigen::Matrix<double,3,1> temp;

            //temp = inv_cov * d_mean;

            //double d_temp = d_mean.transpose()*temp;

            double mhlnbs =  sqrt (d_mean.transpose()*inv_cov * d_mean);

            mahalanobis_dis += mhlnbs;

            // update mahalanobis value of curresponding centroid
            mhln_per_ndt[_it] = mhln_per_ndt[_it] + mhlnbs;

            // update mahalanobis count of curresponding centroid
            ++npoint_per_ndt[_it];
            ++mahalanobis_dis_count;

            // sum of euclidean distance of each point
            euclidean_dis += sqrt (pow(d_mean(0),2) + pow(d_mean(1),2) + pow(d_mean(2),2));
            ++euclidean_dis_count;

        }

        /*

        //  mahalanobis = mahalanobis / point clount
        //  sum_mahalanobis = sum_all(mahalanobis)

        double sum_mahalanobis = 0.0;

        for (int i=0; i< mhln_per_ndt.size(); i++)
        {
            if (npoint_per_ndt[i] == 0)
            {
                mhln_per_ndt[i] = 0.0;
            }
            else
            {
                mhln_per_ndt[i] /= (double)npoint_per_ndt[i];
            }

            sum_mahalanobis += mhln_per_ndt[i];
        }

        double mahalanobis_dis_av = sum_mahalanobis / (double)mhln_per_ndt.size();*/

        double sum_mahalanobis = 0.0;

        for (int i=0; i< mhln_per_ndt.size(); i++)
        {
            sum_mahalanobis += mhln_per_ndt[i];
        }

        double mahalanobis_dis_av = sum_mahalanobis / (double)mahalanobis_dis_count;

        fprintf(pFile, "%f,", (float)mahalanobis_dis_av);

        std::cout << "mahalanobis_dis_av is : " << mahalanobis_dis_av << std::endl;

        //double euclidean_dis_av = euclidean_dis / (double)dirty_centroids.size();
        double euclidean_dis_av = euclidean_dis / (double)euclidean_dis_count;

        //fprintf(pFile, "%f,", (float)euclidean_dis);
        fprintf(pFile, "%f", (float)euclidean_dis_av);

        std::cout << "euclidean_dis_av is : " << euclidean_dis_av << std::endl;

        if (true)
        {
            // get min and max

            double max_mhlndis = -10000.0;
            double min_mhlndis = 10000.0;

            fprintf (pFileLog, "[mahalanobis distance for each centroid]\n");

            for (int i=0; i< mhln_per_ndt.size(); i++)
            {
                mhln_per_ndt[i] = mhln_per_ndt[i] / (double)npoint_per_ndt[i];

                fprintf (pFileLog, "(%i, ", npoint_per_ndt[i]);
                fprintf (pFileLog, "%f) ", mhln_per_ndt[i]);

                if (mhln_per_ndt[i] > max_mhlndis)
                    max_mhlndis = mhln_per_ndt[i];

                if (mhln_per_ndt[i] < min_mhlndis)
                    min_mhlndis = mhln_per_ndt[i];
            }

            fprintf (pFileLog, "[mahalanobis_dis average for local vicinity]\n%f\n", mahalanobis_dis_av);

            // normalize all values

            for (int i=0; i< mhln_per_ndt.size(); i++)
            {
                double temp;
                double diff = max_mhlndis - min_mhlndis;

                if (diff != 0)
                {
                    temp = mhln_per_ndt[i] - min_mhlndis;
                    mhln_per_ndt[i] = temp / diff;
                }
                else
                {
                    mhln_per_ndt[i] = 0.0;
                }

                std::cout <<  mhln_per_ndt[i] << ", ";

            }

            std::cout << std::endl;




            Eigen::Vector4d RGBA(0.0, 0.0, 0.0, 0.0);

            visualization_msgs::MarkerArray dirty_SphereList;

            setCovarianceListMarker_withOpacity<pcl::VoxelGridCovariance<pcl::PointXYZI>::Leaf>\
                    (dirty_leafList, dirty_SphereList, 4.605 , RGBA, mhln_per_ndt, "map", 1000);

            mahlnbs_dis_ndpub.publish(dirty_SphereList);
        }


    }

    std::cout << "test2" << std::endl;


    // show cloud with intensity

    /*
    if (true)
    {
        for (i=0; i< map_velorange.size(); i++)
        {
            pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;

            typename pcl::PointCloud<pcl::PointXYZI>::Ptr centroid_cloud_ptr\
                    (new pcl::PointCloud<pcl::PointXYZI> (centroid_cloud)) ;

            kdtree.setInputCloud (centroid_cloud_ptr);

            // Neighbors within radius search

            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;

            pcl::PointXYZI search_point;
            search_point.x = dirty_ndt[i].centroid.x;
            search_point.y = dirty_ndt[i].centroid.y;
            search_point.z = dirty_ndt[i].centroid.z;

            double manhattan_dis = 0.0; // manhattan distance
            double bhatt_d = 0.0; // bhatacharyya distance

            if ( kdtree.radiusSearch (search_point, ndt_grid_size * 1.3, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
            {
        }
    }*/



    fprintf(pFile, "\n");
    fclose(pFile);

}

int main(int argc, char **argv)
{
    std::cout << "ndt_3d_mapmatching_evaluate_factors_v1_2\n" ;
    ros::init(argc, argv, "ndt_3d_mapmatching_evaluate_factors_v1_2");


    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    skipSeq = 0;

    // Default values
    int iter = 30; // Maximum iterations

    double step_size = 0.1; // Step size
    double trans_eps = 0.01; // Transformation epsilon

    private_nh.getParam("skipSeq", skipSeq);
    private_nh.getParam("generation_interval", generation_interval);
    private_nh.getParam("use_velo_view", use_velo_view);

    private_nh.getParam("calculate_mahalanobis_factor", calculate_mahalanobis_factor);

    private_nh.getParam("x_startpoint", x_startpoint);
    private_nh.getParam("y_startpoint", y_startpoint);
    private_nh.getParam("z_startpoint", z_startpoint);
    private_nh.getParam("roll_startpoint", roll_startpoint);
    private_nh.getParam("pitch_startpoint", pitch_startpoint);
    private_nh.getParam("yaw_startpoint", yaw_startpoint);

    roll_startpoint = (roll_startpoint /180.0) * M_PI;
    pitch_startpoint = (pitch_startpoint /180.0) * M_PI;
    yaw_startpoint = (yaw_startpoint /180.0) * M_PI;

    private_nh.getParam("ndt_grid_size", ndt_grid_size);
    private_nh.getParam("ndt_search_res", ndt_search_res);
    private_nh.getParam("voxel_leaf_size", voxel_leaf_size);


    private_nh.getParam("show_scan", show_scan);
    private_nh.getParam("show_filtered_scan", show_filtered_scan);
    private_nh.getParam("show_transformed_scan", show_transformed_scan);
    private_nh.getParam("show_initial_scan", show_initial_scan);
    private_nh.getParam("show_map", show_map);
    private_nh.getParam("show_car_trajectory", show_car_trajectory);
    private_nh.getParam("save_transformed_scan", save_transformed_scan);
    private_nh.getParam("map_file_path", map_file_path);
    private_nh.getParam("map_file_name", map_file_name);

    private_nh.getParam("map_load_mode", map_load_mode);
    private_nh.getParam("save_predicted_scan", save_predicted_scan);
    private_nh.getParam("save_aligned_scan", save_aligned_scan);
    private_nh.getParam("save_path", save_path);
    private_nh.getParam("save_transformed_dis_scan", save_transformed_dis_scan);
    private_nh.getParam("lidar_range", lidar_range);
    private_nh.getParam("in_vel_topic", in_vel_topic);

    // tf_lidar
    private_nh.getParam("vel_x", vel_x);
    private_nh.getParam("vel_y", vel_y);
    private_nh.getParam("vel_z", vel_z);
    private_nh.getParam("vel_roll", vel_roll);
    private_nh.getParam("vel_pitch", vel_pitch);
    private_nh.getParam("vel_yaw", vel_yaw);

    // load ground truth from file if groundtruth_as_prediction
    private_nh.getParam("groundtruth_as_prediction", groundtruth_as_prediction);
    private_nh.getParam("ground_truth_filename", ground_truth_filename);

    // downsample map or not
    private_nh.getParam("downsample_map", downsample_map);
    private_nh.getParam("map_ds_size", map_ds_size);





    // publish and subscribe topics

    map_pub = nh.advertise<sensor_msgs::PointCloud2>("/map", 1000);
    initial_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/initial_scan", 1000);
    transformed_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/transformed_scan", 1000);
    calibrated_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/calibrated_scan", 1000);
    scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/scan", 1000);
    predicted_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/predicted_scan", 1000);
    filtered_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/filtered_scan", 1000);
    aligned_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned_scan", 1000);
    transformed_dis_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/transformed_dis_scan", 1000);
    dirty_centroids_pub = nh.advertise<sensor_msgs::PointCloud2>("/dirty_centroids", 1000);
    new_dirty_centroids_pub = nh.advertise<sensor_msgs::PointCloud2>("/new_dirty_centroids", 1000);

    map_velorange_pub = nh.advertise<sensor_msgs::PointCloud2>("/map_velorange", 1000);
    map_veloview_pub = nh.advertise<sensor_msgs::PointCloud2>("/map_veloview", 1000);

    ground_dirty_ndt_pub = nh.advertise<sensor_msgs::PointCloud2>("/ground_dirty_ndt", 1000);
    nonground_dirty_ndt_pub = nh.advertise<sensor_msgs::PointCloud2>("/nonground_dirty_ndt", 1000);
    normal_histogram_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/normal_histogram_cloud", 1000);

    odom_pub = nh.advertise<nav_msgs::Odometry>("/ndt_3d_mapmatching/odom", 100);
    ndt_map_pub = nh.advertise<visualization_msgs::MarkerArray>("ndt_map", 10000);
    dimension_ndt_pub = nh.advertise<visualization_msgs::MarkerArray>("dimension_ndt", 10000);
    dirty_ndt_pub = nh.advertise<visualization_msgs::MarkerArray>("dirty_ndt", 10000);
    pfh_sim_ndpub = nh.advertise<visualization_msgs::MarkerArray>("pfh_sim_nd_pub", 10000);
    pfh_manh_ndpub = nh.advertise<visualization_msgs::MarkerArray>("pfh_manh_nd_pub", 10000);
    bhat_dis_ndpub = nh.advertise<visualization_msgs::MarkerArray>("bhat_dis_ndpub", 10000);
    mahlnbs_dis_ndpub = nh.advertise<visualization_msgs::MarkerArray>("mahlnbs_dis_ndpub", 10000);

    vis_pub = nh.advertise<visualization_msgs::Marker>( "vehicle", 10 );


    // for test
    pfh_similarity_entropy_pub = nh.advertise<sensor_msgs::PointCloud2>("/pfh_similarity_entropy", 1000);
    pfh_similarity_manhattan_pub = nh.advertise<sensor_msgs::PointCloud2>("/similarity_manhattan", 1000);
    lat_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/lat_weight", 1000);
    long_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/long_weight", 1000);
    normal_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/normal_cloud", 1000);
    depth_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/occupancy_cloud", 1000);



    ros::Subscriber scan_sub = nh.subscribe(in_vel_topic, 12000, scan_callback);


    // get groundtruth

    if (groundtruth_as_prediction)
    {
        std::ifstream  data(ground_truth_filename);
        if (!data.is_open())
        {
            ROS_ERROR("\n\nError opening groundtruth pose listfor \n%s", ground_truth_filename);
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

            // 0 --> seq
            std::getline(lineStream,cell,',');
            pose_entity_temp.seq = atof(cell.c_str());

            // 1 --> time
            std::getline(lineStream,cell,',');
            pose_entity_temp.stamp = atof(cell.c_str());

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

            ground_truth_list.push_back(pose_entity_temp);

        }

    }

    // make necessary folders

    // make some string for file naming

    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << ndt_grid_size;
    std::string s = stream.str();

    std::stringstream stream2;
    stream2 << std::fixed << std::setprecision(1) << ndt_search_res;
    std::string s2 = stream2.str();

    std::stringstream stream3;
    stream3 << std::fixed << std::setprecision(1) << voxel_leaf_size;
    std::string s3 = stream3.str();

    std::stringstream stream4;
    stream4 << std::fixed << std::setprecision(1) << lidar_range;
    std::string s4 = stream4.str();


    std::string name_tail = "nd(" + s + ")_ds(" + s3 + ")_sr(" + s2 + ")_lidar(" + s4 + "m)";


    time_t timer;
    time(&timer);

    std::stringstream ss;
    ss << timer;
    std::string str_time = ss.str();

    struct stat st = {0};

    if (stat(save_path.c_str(), &st) == -1)
    {
        mkdir(save_path.c_str(), 0700);
    }

    save_path = save_path + str_time + "_factors_" + name_tail + "/";

    if (stat(save_path.c_str(), &st) == -1)
    {
        mkdir(save_path.c_str(), 0700);
    }

    FILE* pFile;

    std::string  name = save_path + "vehicle_pose.csv";
    pFile = fopen (name.c_str(), "w");
    fclose (pFile);

    name = save_path + "factors.csv";
    pFile = fopen (name.c_str(), "w");
    fclose (pFile);


    // make new files because later we add to the file

    name = save_path + "vehicle_pose.csv";



    pFile = fopen(name.c_str(), "a");

    fprintf (pFile, "seq,time,x,y,z,roll,pitch,yaw,converged,fitnessScore,trans_prob,iter,downsample_t,align_t,matching_t,scan_size,trans_scan_size,init_err,init_yaw_err\n");

    fclose(pFile);

    name = save_path + "factors.csv";

    pFile = fopen(name.c_str(), "a");

    // cent_count_view is actual centroids that can be captured by laser scanner

    fprintf (pFile, "seq,time,x,y,z,occupancy_ratio,cent_count,cent_count_view,nongr_cent_count,D1cnt,D2cnt,D3cnt,");
    fprintf (pFile, "D1cntRate,D2cntRate,D3cntRate,D1valAve,D2valAve,D3valAve,long_weight,lat_weight,");
    fprintf (pFile, "r_ave,nongr_D2_count,PDOP,D2weight,D2weight_with_eigen,D2weightAve,D2weightWithEigenAve,");
    fprintf (pFile, "layout_ang_H,layout_ang_H_1D2D,normal_H_90,normal_H_16,normal_H_16_2D,normal_H_8,");
    fprintf (pFile, "pfh_sim_H,pfh_sim_mnht,bhatt_sim,");
    fprintf (pFile, "mhlnb_disAve,eucl_disAve\n");

    fclose(pFile);

    // log file

    name = save_path + "evaluate_factor.log";
    pFileLog = fopen(name.c_str(), "a");
    fprintf (pFileLog, "log file for evaluate factor\n");



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


    // load map

    if (!load_pointcloud_map<PointXYZI>(map_file_path.c_str(), map_file_name.c_str(), map_load_mode, map_ptr))
    {
        std::cout << "error occured while loading map from following path :"<< std::endl;
        //std::cout << "map_file_path << std::endl;
    }
    else
    {
        map_loaded = true;
    }

    std::cout << "loaded map with point size : " << map_ptr->size() << std::endl;
    map_loaded = true;

    // downsample map

    pcl::PointCloud<PointXYZI> filtered_map;

    //pcl::ApproximateVoxelGrid<PointXYZI> voxel_grid_filter;
    pcl::VoxelGrid<PointXYZI> voxel_grid_filter;
    voxel_grid_filter.setLeafSize(map_ds_size, map_ds_size, map_ds_size);
    voxel_grid_filter.setInputCloud(map_ptr);
    voxel_grid_filter.filter(filtered_map);

    filtered_map_ptr = (pcl::PointCloud<pcl::PointXYZI>::Ptr)(&filtered_map);

    if (downsample_map)
    {
        map_ptr = ( pcl::PointCloud<PointXYZI>::Ptr)(&filtered_map);
        std::cout << "Use down-sampled map with size of : " << filtered_map.size() << std::endl;
    }





    // Setting NDT parameters to default values

    ndt.setMaximumIterations(iter);
    ndt.setResolution(ndt_grid_size);
    ndt.setStepSize(step_size);
    ndt.setTransformationEpsilon(trans_eps);
    ndt.setSearchResolution(ndt_search_res);

    // Setting point cloud to be aligned to.
    ndt.setInputTarget(map_ptr);


    // SHOW MAP IN RVIZ #####################################################################################

    if (show_map)
    {
        //std::cout << "Show filtered map with size of : " << filtered_map.size() << std::endl;

        publish_pointCloud(*map_ptr, map_pub, "map");
        //publish_pointCloud(filtered_map, map_pub, "map");

        // get ndt map

        pcl::VoxelGridCovariance<pcl::PointXYZI> target_cells;
        ndt.getCells(target_cells);

        //pcl::VoxelGridCovariance<pcl::PointXYZ>::Leaf leaf_;
        //std::map<size_t, pcl::VoxelGridCovariance<pcl::PointXYZI>::Leaf> leaves;
        //leaves = target_cells.getLeaves();
        //std::vector<Leaf> leafList;
        //getLeaves(leafList);

        typedef pcl::VoxelGridCovariance<pcl::PointXYZI> VectorCovarianceXYZ;
        typedef typename VectorCovarianceXYZ::Leaf VectorCovarianceLeaf;
        typedef std::vector<VectorCovarianceLeaf> VectorCovarianceLeafList;

        VectorCovarianceLeafList leafList;
        ndt.getLeaves(leafList);

        visualization_msgs::MarkerArray dim_ndt_list;
        visualization_msgs::MarkerArray ndt_list;



        //double d1 = ndt.get_d1();
        //double d2 = ndt.get_d2();

        // a 90% confidence interval corresponds to scale=4.605
        //showCovariance(leaves, ndt_list, 4.605 ,"map", RGBA, d1, d2);
        //showCovariance(leaves, ndt_list, 4.605 ,"map", normalDistribution_color);


        setCovarianceListMarker<pcl::VoxelGridCovariance<pcl::PointXYZI>::Leaf>(leafList, dim_ndt_list, \
                                                      4.605 ,"map", 20);

        dimension_ndt_pub.publish(dim_ndt_list);


        //Eigen::Vector4d RGBA(0.35, 0.7, 0.8, 0.2);
        Eigen::Vector4d normalDistribution_color(0.35, 0.7, 0.8, 0.4);
        setCovarianceListMarker<pcl::VoxelGridCovariance<pcl::PointXYZI>::Leaf>(leafList, ndt_list, \
                                                      4.605 ,"map", normalDistribution_color, 2000);

        ndt_map_pub.publish(ndt_list);


        show_map = 0;
    }

    // load vehicle model

    car.header.frame_id = "/map";
    car.header.stamp = ros::Time();
    car.ns = "car";
    car.id = 21;
    car.type = visualization_msgs::Marker::MESH_RESOURCE;
    car.mesh_resource = "package://ndt_3d_mapmatching/Mercedes_G.dae";
    car.action = visualization_msgs::Marker::ADD;
    //car.scale(1.0, 1.0, 1.0);
    car.scale.x = 0.65;
    car.scale.y = 0.65;
    car.scale.z = 0.65;
    car.pose.position.x = 0.0;
    car.pose.position.y = 0.0;
    car.pose.position.z = -2.4;
    car.pose.orientation.x = 0.0;
    car.pose.orientation.y = 0.0;
    car.pose.orientation.z = 180.0;
    car.pose.orientation.w = 1.0;
    //car.lifetime = ;
    car.frame_locked = false;
    car.text = "This is some text\nthis is a new line\nthis ";
    car.mesh_use_embedded_materials = true;
    car.color.a = 1.0; // Don't forget to set the alpha!
    car.color.r = 0.0;
    car.color.g = 0.3;
    car.color.b = 1.0;

    vis_pub.publish(car);

    ros::spin();


    fclose(pFileLog);
    return 0;
}

