/*
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
#include <kmj_point_type/point_types.h>
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

#include "kmj_ndt/visual_ndt.h"
#include "kmj_ndt/show_ndt.h"
#include "kmj_entropy/entropy.h"
#include "kmj_common/kmj_common.h"
#include "kmj_point_type/self_driving_point_type.h"
#include "kmj_map/kmj_map_pcd.h"


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



// Leaf size of VoxelGrid filter.

double voxel_leaf_size = 1.0;

// publishers

ros::Publisher map_pub;
ros::Publisher ndt_map_pub;
ros::Publisher transformed_scan_pub;
ros::Publisher initial_scan_pub;
ros::Publisher calibrated_scan_pub;
ros::Publisher scan_pub;
ros::Publisher predicted_scan_pub;
ros::Publisher filtered_scan_pub;
ros::Publisher carTrajectory_pub;
ros::Publisher aligned_scan_pub;
ros::Publisher transformed_dis_scan_pub;
ros::Publisher origin_dirty_centroids_pub;
ros::Publisher dirty_centroids_pub;
ros::Publisher odom_pub;


// show data on rviz
bool show_scan = true;
bool show_filtered_scan = true;
bool show_transformed_scan = true;
bool show_initial_scan = true;
bool show_map = true;
bool show_car_trajectory = true;
bool show_transformed_dis_scan = true;



// save scan data
bool save_input_scan = false;
bool save_transformed_scan = false;
bool save_predicted_scan = false;
bool save_aligned_scan = false;
bool save_transformed_dis_scan = false;

std::string map_file_path; // should be null otherwise it will have error
std::string map_file_name;
std::string save_path = "/home/ehsan/temp/results/map_matching/";
int map_load_mode = 0;

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
};

Eigen::Matrix4f tf_ground_truth;
pose ground_truth_pose;
pose previous_ground_truth_pose;
std::vector<pose_entity> ground_truth_list;


FILE * pFileLog;

pcl::PCDWriter writer;

pcl::PointCloud<PointXYZI>::Ptr map_ptr (new pcl::PointCloud<PointXYZI>);

// these tow variable is for log file only to show where the point cloud is saved
std::string savedMap ="";
std::string savedRoadMarkingWindow = "";

std::vector<pose> carPoseList;
double lidar_range = 100.0;
std::string in_vel_topic = "/topic_preprocessor/vel_scan";

double vel_x, vel_y, vel_z, vel_roll, vel_pitch, vel_yaw = 0.0;


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

    // SHOW MAP IN RVIZ #####################################################################################

    if (show_map)
    {
        publish_pointCloud(*map_ptr, map_pub, "map");

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

        visualization_msgs::MarkerArray ndtSphereList;
        //Eigen::Vector4d RGBA(0.35, 0.7, 0.8, 0.2);
        Eigen::Vector4d normalDistribution_color(0.35, 0.7, 0.8, 0.4);

        //double d1 = ndt.get_d1();
        //double d2 = ndt.get_d2();

        // a 90% confidence interval corresponds to scale=4.605
        //showCovariance(leaves, ndtSphereList, 4.605 ,"map", RGBA, d1, d2);
        //showCovariance(leaves, ndtSphereList, 4.605 ,"map", normalDistribution_color);
        setCovarianceListMarker<pcl::VoxelGridCovariance<pcl::PointXYZI>::Leaf>(leafList, ndtSphereList, \
                                                      4.605 ,"map", 20);

        //setCovarianceListMarker<pcl::VoxelGridCovariance<pcl::PointXYZI>::Leaf>(leafList, ndtSphereList, \
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

    scan.height = 1;
    scan.width = scan.size();
    scan.points.resize (scan.width * scan.height);

    // save input scan without any transformation

    if (save_input_scan)
    {
        std::string name = save_path + "scan_" + \
                           std::to_string((input->header.seq)) + ".pcd";

        writer.write(name, scan, false);
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
        // EXTRACT CURRENT PREDICTION FROM PREDICTED POSE LIST ##################################################

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

    carPoseList.push_back(current_pose);

    align_end = std::chrono::system_clock::now();

    alignTime = std::chrono::duration_cast<std::chrono::microseconds>\
                                        (align_end - align_start).count()/1000.0;

    matchingTime = alignTime + downSampleTime;

    // publihs odom

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

    std::cout << "DownSampleTime : " << downSampleTime << std::endl;
    std::cout << "AlignTime : " << alignTime << std::endl;
    std::cout << "MatchingTime : " << matchingTime << std::endl;
    std::cout << "Number of iteration : " << ndt.getFinalNumIteration() << std::endl;
    std::cout << "trans_probability : " << trans_probability << std::endl;
    std::cout << "Size of input points after downsampling : " << input_point_size << std::endl;

    std::cout << "3D Displacement (current - previous) in meter " << displacement_3d << std::endl;
    std::cout << "Error of initial guess (3D) in meter " << initial_guess_error << std::endl;
    std::cout << "Yaw error of initial guess in degree " << initial_guess_yaw_error << std::endl;


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

        std::string name = save_path + "map_matching_log.csv";

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


}

int main(int argc, char **argv)
{
    std::cout << "ndt_3d_mapmatching_v5_2\n" ;
    ros::init(argc, argv, "ndt_3d_mapmatching_v5_2");


    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    skipSeq = 0;

    double ndt_search_res = 1.0;
    double ndt_grid_size = 1.0;

    // Default values
    int iter = 30; // Maximum iterations

    double step_size = 0.1; // Step size
    double trans_eps = 0.01; // Transformation epsilon

    private_nh.getParam("x_startpoint", x_startpoint);
    private_nh.getParam("y_startpoint", y_startpoint);
    private_nh.getParam("z_startpoint", z_startpoint);
    private_nh.getParam("roll_startpoint", roll_startpoint);
    private_nh.getParam("pitch_startpoint", pitch_startpoint);
    private_nh.getParam("yaw_startpoint", yaw_startpoint);

    roll_startpoint = (roll_startpoint /180.0) * M_PI;
    pitch_startpoint = (pitch_startpoint /180.0) * M_PI;
    yaw_startpoint = (yaw_startpoint /180.0) * M_PI;

    // path
    private_nh.getParam("save_path", save_path);
    private_nh.getParam("map_file_path", map_file_path);
    private_nh.getParam("map_file_name", map_file_name);

    // ndt parameters
    private_nh.getParam("ndt_grid_size", ndt_grid_size);
    private_nh.getParam("ndt_search_res", ndt_search_res);
    private_nh.getParam("voxel_leaf_size", voxel_leaf_size);

    private_nh.getParam("show_scan", show_scan);
    private_nh.getParam("show_filtered_scan", show_filtered_scan);
    private_nh.getParam("show_transformed_scan", show_transformed_scan);
    private_nh.getParam("show_initial_scan", show_initial_scan);
    private_nh.getParam("show_map", show_map);
    private_nh.getParam("show_car_trajectory", show_car_trajectory);

    // map load mode
    private_nh.getParam("map_load_mode", map_load_mode);

    // save pcd files
    private_nh.getParam("save_input_scan", save_input_scan);
    private_nh.getParam("save_predicted_scan", save_predicted_scan);
    private_nh.getParam("save_aligned_scan", save_aligned_scan);
    private_nh.getParam("save_transformed_scan", save_transformed_scan);
    private_nh.getParam("save_transformed_dis_scan", save_transformed_dis_scan);

    // lidar parameters
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

            ground_truth_list.push_back(pose_entity_temp);

        }
    }



    time_t timer;
    time(&timer);

    std::stringstream ss;
    ss << timer;
    std::string str_time = ss.str();

    save_path = save_path + str_time + "/";

    struct stat st = {0};

    if (stat(save_path.c_str(), &st) == -1)
    {
        mkdir(save_path.c_str(), 0700);
    }

    // make new files because later we add to the file

    std::string name = save_path + "map_matching_log.csv";

    FILE *pFile;

    pFile = fopen (name.c_str(), "w");

    fprintf (pFile, " (int)input->header.seq, \
             (float)current_scan_time,\
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
             (float)initial_guess_yaw_error\n");

    fclose (pFile);


    if (!load_pointcloud_map<PointXYZI>(map_file_path.c_str(), map_file_name.c_str(), map_load_mode, map_ptr))
    {
        std::cout << "error occured while loading map from following path :"<< std::endl;
        std::cout << map_file_path << std::endl;
    }
    else
    {
        std::cout << "Map loaded correcly!"<< std::endl;
        map_loaded = true;
    }

    std::cout << map_ptr->size() << std::endl;
    map_loaded = true;

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
    ndt.setMaximumIterations(iter);
    ndt.setResolution(ndt_grid_size);
    ndt.setStepSize(step_size);
    ndt.setTransformationEpsilon(trans_eps);
    ndt.setSearchResolution(ndt_search_res);

    //std::cout << ndt.getSearchResolution() << std::endl;

    // Setting point cloud to be aligned to.
    ndt.setInputTarget(map_ptr);

    map_pub = nh.advertise<sensor_msgs::PointCloud2>("/map", 1000);
    initial_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/initial_scan", 1000);
    transformed_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/transformed_scan", 1000);
    calibrated_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/calibrated_scan", 1000);
    scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/scan", 1000);
    predicted_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/predicted_scan", 1000);
    filtered_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/filtered_scan", 1000);
    aligned_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned_scan", 1000);
    transformed_dis_scan_pub = nh.advertise<sensor_msgs::PointCloud2>("/transformed_dis_scan", 1000);
    origin_dirty_centroids_pub = nh.advertise<sensor_msgs::PointCloud2>("/origin_dirty_centroids", 1000);
    dirty_centroids_pub = nh.advertise<sensor_msgs::PointCloud2>("/dirty_centroids", 1000);
    odom_pub = nh.advertise<nav_msgs::Odometry>("/ndt_3d_mapmatching/odom", 100);



    ndt_map_pub = nh.advertise<visualization_msgs::MarkerArray>("ndt_map", 10000);
    carTrajectory_pub = nh.advertise<visualization_msgs::Marker>("CarTrajectory3D_line", 1000);

    ros::Subscriber scan_sub = nh.subscribe(in_vel_topic, 1000, scan_callback);


    publish_pointCloud(*map_ptr, map_pub, "map");




    ros::spin();



    return 0;
}
