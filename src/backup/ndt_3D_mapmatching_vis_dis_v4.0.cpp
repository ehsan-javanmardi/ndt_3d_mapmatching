/*


 Localization program using distortion removed visual Normal Distributions Transform
 Matching scan over point cloud map



 Ehsan Javanmardi

 2017.07.10
 ndt_3D_mapmatching_vis_dis_v4.0

 TETS :
    Code without Distortion removal works well
    But with distortion removal it fails most of the time

    Entropy part has some problems that I didn;t correct it yet!!
    For next version I will deal with this (v4.1)


 CHANGE LOG :
    Based on based on ndt_3D_mapmatching_vis_dis_v3.1
    Use new point type PointXYZIT doe delta_t and distortion
    Point type is defined in self_driving_point_type.h
    Report more evaluation parameters

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

// Visual NDT with distortion removal
NormalDistributionsTransform_Visual_dis<PointXYZIT, PointXYZIT> ndt;

// Default values
static int iter = 30; // Maximum iterations
static double gridSize = 1.0; // Resolution
static double step_size = 0.1; // Step size
static double trans_eps = 0.01; // Transformation epsilon

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


// show data on rviz

bool show_scan = true;
bool show_filtered_scan = true;
bool show_transformed_scan = true;
bool show_initial_scan = true;
bool show_map = true;
bool show_car_trajectory = true;
bool show_transformed_dis_scan = true;

bool do_distortion_removal = true;

// save scan data

bool save_transformed_scan = false;
bool save_predicted_scan = false;
bool save_aligned_scan = false;
bool save_transformed_dis_scan = false;

std::string map_file_path;
std::string map_file_name;
std::string save_path = "/home/ehsan/workspace/results/map_matching/";
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

static double x_startpoint = 0.0;
static double y_startpoint = 0.0;
static double z_startpoint = 0.0;
static double yaw_startpoint =  0.0;//(-45/180.0) * M_PI ;
static double roll_startpoint =  0.0;//(0/180.0) * M_PI ;
static double pitch_startpoint = 0.0;//(-33/180.0) * M_PI  ;

FILE * pFileLog;

pcl::PCDWriter writer;

pcl::PointCloud<PointXYZIT>::Ptr map_ptr (new pcl::PointCloud<PointXYZIT>);

// these tow variable is for log file only to show where the point cloud is saved
std::string savedMap ="";
std::string savedRoadMarkingWindow = "";

std::vector<pose> carPoseList;

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
        Eigen::Vector4d normalDistribution_color(0.35, 0.7, 0.8, 0.2);

        //double d1 = ndt.get_d1();
        //double d2 = ndt.get_d2();

        // a 90% confidence interval corresponds to scale=4.605
        //showCovariance(leaves, ndtSphereList, 4.605 ,"map", RGBA, d1, d2);
        //showCovariance(leaves, ndtSphereList, 4.605 ,"map", normalDistribution_color);
        setCovarianceListMarker<pcl::VoxelGridCovariance<PointXYZIT>::Leaf>(leafList, ndtSphereList, \
                                                      4.605 ,"map", 20);

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
    static Eigen::Matrix4f tf_lidar;

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

    // SHOW PREDICTED SCAN ###################################################################

    if (show_scan)
    {
        pcl::PointCloud<PointXYZIT>  predicted_scan;

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

    if (do_distortion_removal)
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

    // ALIGN POINT CLOUD TO THE MAP ##################################################################

    align_start = std::chrono::system_clock::now();

    //pcl::PointCloud<pcl::PointXYZI>::Ptr scan_ptr(new pcl::PointCloud<pcl::PointXYZI>(scan));

    pcl::PointCloud<PointXYZIT>::Ptr filter_scan_ptr(new pcl::PointCloud<PointXYZIT>(filtered_scan));

    ndt.setInputSource(filter_scan_ptr);

    if (do_distortion_removal)
        ndt.setPreviousTF(tf_previous);

    ndt.setSearchResolution(gridSize);

    pcl::PointCloud<PointXYZIT> aligned_scan;

    // aligned scan is distortion removed downsampled scan so it is not useful

    ndt.align(aligned_scan, tf_predict);

    // get the global translation matrix

    Eigen::Matrix4f tf_align(Eigen::Matrix4f::Identity()); // base_link
    tf_align = ndt.getFinalTransformation(); // localizer

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


    // show dirty centroids
/*
    if (true)
    {
        pcl::PointCloud<PointXYZINormal> dirty_centroids;

        std::vector<Eigen::Vector3d> centroids_Dimentoin_shapes;

        ndt.getDirtyCentroidsWithNormal(dirty_centroids, centroids_Dimentoin_shapes, *filter_scan_ptr, tf_align);

        pcl::PointCloud<pcl::PointXYZI> points;

        for (int i=0; i< dirty_centroids.size(); i++)
        {
            pcl::PointXYZI p;

            p.x = dirty_centroids[i].x;
            p.y = dirty_centroids[i].y;
            p.z = dirty_centroids[i].z;
            p.intensity = dirty_centroids[i].curvature;

            points.push_back(p);
        }

        publish_pointCloud(points, dirty_centroids_pub, "map");
    }

    // show original centroids (moved centroids to the 0,0, coordinate)

    if (true)
    {
        pcl::PointCloud<PointXYZINormal> dirty_centroids;

        std::vector<Eigen::Vector3d> centroids_Dimentoin_shapes;

        ndt.getDirtyCentroidsWithNormal(dirty_centroids, centroids_Dimentoin_shapes, *filter_scan_ptr, tf_align);

        transformPointCloudToOrigin(dirty_centroids, dirty_centroids, tf_align);

        pcl::PointCloud<pcl::PointXYZI> points;

        for (int i=0; i< dirty_centroids.size(); i++)
        {
            pcl::PointXYZI p;

            p.x = dirty_centroids[i].x;
            p.y = dirty_centroids[i].y;
            p.z = dirty_centroids[i].z;
            p.intensity = dirty_centroids[i].curvature;

            points.push_back(p);
        }

        publish_pointCloud(points, origin_dirty_centroids_pub, "map");

    }


    // get layout histogram for only non ground

    double layout_non_ground_entropy;

    if (true)
    {
        pcl::PointCloud<PointXYZINormal> dirty_centroids;

        std::vector<Eigen::Vector3d> centroids_Dimentoin_shapes;

        ndt.getDirtyCentroidsWithNormal(dirty_centroids, centroids_Dimentoin_shapes, *filter_scan_ptr, tf_align);

        int num_bin = 36;
        int histogram[num_bin]={};

        double min_bin_val = 0.0;
        double max_bin_val = 360.0;

        double ground_height = tf_align(2,3) - 2.0;

        if (makeLayoutHistogram(dirty_centroids, histogram,\
                            num_bin, min_bin_val, max_bin_val, \
                            ground_height, tf_align) != 0)
        {
            PCL_ERROR("ERROR in FUNCTION makeLayoutHistogram\n");
        }

        // save histogram

        std::string name = save_path + "layout_histogram_nonground.csv";

        FILE* pFile;

        pFile = fopen(name.c_str(), "a");

        for (int i=0; i<num_bin; i++)
        {
            fprintf (pFile, "%i,", histogram[i]);
        }

        calculateEntropyFormHistogram(histogram, num_bin, layout_non_ground_entropy);

        fprintf (pFile, "%f,", layout_non_ground_entropy);

        fprintf (pFile, "\n");

        fclose(pFile);
    }

    double layout_all_entropy;

    // get layout histogram for all points (non ground + ground points)

    if (true)
    {
        pcl::PointCloud<PointXYZINormal> dirty_centroids;

        std::vector<Eigen::Vector3d> centroids_Dimentoin_shapes;

        ndt.getDirtyCentroidsWithNormal(dirty_centroids, centroids_Dimentoin_shapes, *filter_scan_ptr, tf_align);

        int num_bin = 36;
        int histogram[num_bin]={};

        double min_bin_val = 0.0;
        double max_bin_val = 360.0;

        double ground_height = -10.0;

        if (makeLayoutHistogram(dirty_centroids, histogram,\
                            num_bin, min_bin_val, max_bin_val, \
                            ground_height, tf_align) != 0)
        {
            PCL_ERROR("ERROR in FUNCTION makeLayoutHistogram\n");
        }

        // save histogram

        std::string name = save_path + "layout_histogram_all.csv";

        FILE* pFile;

        pFile = fopen(name.c_str(), "a");

        for (int i=0; i<num_bin; i++)
        {
            fprintf (pFile, "%i,", histogram[i]);
        }

        calculateEntropyFormHistogram(histogram, num_bin, layout_all_entropy);

        fprintf (pFile, "%f,", layout_all_entropy);

        fprintf (pFile, "\n");

        fclose(pFile);

    }


    // Get histogram of dimention shape of points





    // Get lateral and long features

    double x_points_feature_entropy;
    double y_points_feature_entropy;

    if (true)
    {
        pcl::PointCloud<PointXYZINormal> dirty_centroids;
        pcl::PointCloud<PointXYZINormal> tr_centroids;

        std::vector<Eigen::Vector3d> centroids_Dimentoin_shapes;

        ndt.getDirtyCentroidsWithNormal(dirty_centroids, centroids_Dimentoin_shapes, *filter_scan_ptr, tf_align);

        int num_bin = 200;
        int x_histogram[num_bin]={};
        int y_histogram[num_bin]={};
        double min_bin_val = -100.0;
        double max_bin_val = 100.0;
        double ground_height = -10.0;

        makeXYPointsHistogramInOrigin(dirty_centroids, x_histogram, y_histogram,
                                      num_bin, min_bin_val, max_bin_val, ground_height,
                                      tf_align, tr_centroids);

        // save histograms

        std::string name = save_path + "x_points_histogram.csv";

        FILE* pFile;

        pFile = fopen(name.c_str(), "a");

        for (int i=0; i<num_bin; i++)
        {
            fprintf (pFile, "%i,", x_histogram[i]);
        }

        calculateEntropyFormHistogram(x_histogram, num_bin, x_points_feature_entropy);

        fprintf (pFile, "%f,", x_points_feature_entropy);

        fprintf (pFile, "\n");

        fclose(pFile);



        std::string name2 = save_path + "y_points_histogram.csv";

        FILE* pFile2;

        pFile2 = fopen(name2.c_str(), "a");

        for (int i=0; i<num_bin; i++)
        {
            fprintf (pFile2, "%i,", y_histogram[i]);
        }

        calculateEntropyFormHistogram(y_histogram, num_bin, y_points_feature_entropy);

        fprintf (pFile2, "%f,", y_points_feature_entropy);

        fprintf (pFile2, "\n");

        fclose(pFile2);

    }




    double yaw_ndt_normal_diff_entropy;

    if (true)
    {
        pcl::PointCloud<PointXYZINormal> dirty_centroids;

        std::vector<Eigen::Vector3d> centroids_Dimentoin_shapes;

        ndt.getDirtyCentroidsWithNormal(dirty_centroids, centroids_Dimentoin_shapes, *filter_scan_ptr, tf_align);

        int num_bin = 20;
        int histogram[num_bin]={};

        double ground_height = tf_align(2,3) - 2.0;

        if (makeYawAndNDTNormalDiffHistogram(dirty_centroids, histogram, num_bin, ground_height, tf_align) != 0)
        {
            PCL_ERROR("ERROR in FUNCTION makeYawAndNDTNormalDiffHistogram\n");
        }

        // save histogram

        std::string name = save_path + "yaw_and_NDT_dif_histogram.csv";

        FILE* pFile;

        pFile = fopen(name.c_str(), "a");

        for (int i=0; i<num_bin; i++)
        {
            fprintf (pFile, "%i,", histogram[i]);
        }

        calculateEntropyFormHistogram(histogram, num_bin, yaw_ndt_normal_diff_entropy);

        fprintf (pFile, "%f,", yaw_ndt_normal_diff_entropy);

        fprintf (pFile, "\n");

        fclose(pFile);

    }*/


    // calculate and save histogram

    // calculate and save entropy


    // getDirtyCentroids normal not redundant

    /*getDirtyCentroidsWithNormalNotRedundant(pcl::PointCloud<PointXYZINormal> &centroids, \
                                                pcl::PointCloud<PointSource> input_cloud,\
                                                Eigen::Matrix4f tf_guess);*/

    // calculate and save histogram

    // calculate and save entropy


    // get layout histogram




    // get points in neighborhood




//    getDirtyCentroidsWithNormal();



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
    std::cout << "Number of centroids : " << ndt.getCentroidsCount() << std::endl << std::endl;
    //std::cout << "Score : " << score << std::endl;
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

    // this cloud is distortion-eliminated

    if (save_aligned_scan)
    {
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

    if (do_distortion_removal)
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

    // SHOW DISTORTION REMOVED TRANSFORMED SCAN ################################################################################

    if (show_transformed_scan)
    {
        publish_pointCloud(transformed_scan, transformed_scan_pub, "map");
    }

    // SAVE DISTORTION REMOVED TRANSFORMED SCAN #################################################################################

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

        std::string name = save_path + "map_matching_log.csv";

        FILE* pFile;
        pFile = fopen(name.c_str(), "a");

  /*      fprintf (pFile, "%i,%f,%f,%f,%f,%f,%f,%i,%f,%f,%i,%f,%f,%f,%f,%i,%i,%f,%f,%f,%f,%f,%f,%f\n",\
                 (int)input->header.seq, \
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
                 //(float)ndt.getScore(),
                 (float)0.0,\
                 (float)downSampleTime,\
                 (float)alignTime,\
                 (float)matchingTime,\
                 (int)scan.size(), \
                 (int)transformed_scan.size(),\
                 (float)initial_guess_error,\
                 (float)initial_guess_yaw_error,\
                 (float)layout_non_ground_entropy,\
                 (float)layout_all_entropy,\
                 (float)x_points_feature_entropy,\
                 (float)y_points_feature_entropy,\
                 (float)yaw_ndt_normal_diff_entropy);*/

        fclose(pFile);

    }


}

int main(int argc, char **argv)
{
    std::cout << "ndt_3D_mapmatching_vis_dis_v4_0\n" ;
    ros::init(argc, argv, "ndt_3D_mapmatching_vis_dis_v4_0");


    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    skipSeq = 0;

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

    if (private_nh.getParam("yaw_startpoint", yaw_startpoint) == false)
    {
        std::cout << "yaw_startpoint is not set." << std::endl;
        //return -1;
    }
    std::cout << "yaw_startpoint in degree : " << yaw_startpoint << std::endl;
    yaw_startpoint = (yaw_startpoint /180.0) * M_PI;
    //yaw_startpoint = ((124.0 - 157.3)/180.0) * M_PI;

    if (private_nh.getParam("pitch_startpoint", pitch_startpoint) == false)
    {
        std::cout << "pitch_startpoint is not set." << std::endl;
        //return -1;
    }
    std::cout << "pitch_startpoint in degree : " << pitch_startpoint << std::endl;
    pitch_startpoint = (pitch_startpoint /180.0) * M_PI;

    if (private_nh.getParam("roll_startpoint", roll_startpoint) == false)
    {
        std::cout << "roll_startpoint is not set." << std::endl;
        //return -1;
    }
    std::cout << "roll_startpoint in degree : " << roll_startpoint << std::endl;
    roll_startpoint = (roll_startpoint /180.0) * M_PI;

    if (private_nh.getParam("gridSize", gridSize) == false)
    {
      std::cout << "gridSize is not set." << std::endl;
      //return -1;
    }
    std::cout << "gridSize: " << gridSize << std::endl;

    if (private_nh.getParam("skipSeq", skipSeq) == false)
    {
      std::cout << "skipSeq is not set." << std::endl;
      //return -1;
    }
    std::cout << "skipSeq: " << skipSeq << std::endl;

    if (private_nh.getParam("voxel_leaf_size", voxel_leaf_size) == false)
    {
      std::cout << "voxel_leaf_size is not set." << std::endl;
      //return -1;
    }
    std::cout << "voxel_leaf_size: " << voxel_leaf_size << std::endl;

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

    if (private_nh.getParam("save_transformed_scan", save_transformed_scan) == false)
    {
      std::cout << "save_transformed_scan is not set." << std::endl;
      //return -1;
    }
    std::cout << "save_transformed_scan: " << save_transformed_scan << std::endl;

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

    if (private_nh.getParam("do_distortion_removal", do_distortion_removal) == false)
    {
      std::cout << "do_distortion_removal is not set." << std::endl;
      //return -1;
    }
    std::cout << "do_distortion_removal: " << do_distortion_removal << std::endl;








    // Make necesssary folders

    if (true)
    {
        struct stat st = {0};

        std::string name = save_path;
        //save_path[save_path.size()] = ' ';

        if (stat(name.c_str(), &st) == -1)
        {
            mkdir(save_path.c_str(), 0700);
        }

        name = save_path + "transformed_dis_scan";

        if (stat(name.c_str(), &st) == -1)
        {
            mkdir(name.c_str(), 0700);
        }

        name = save_path + "transformed_scan";

        if (stat(name.c_str(), &st) == -1)
        {
            mkdir(name.c_str(), 0700);
        }

        name = save_path + "aligned_scan";

        if (stat(name.c_str(), &st) == -1)
        {
            mkdir(name.c_str(), 0700);
        }

        name = save_path + "predicted_scan";

        if (stat(name.c_str(), &st) == -1)
        {
            mkdir(name.c_str(), 0700);
        }

    }

    // make new files because later we add to the file

    std::string name = save_path + "map_matching_log.csv";

    FILE *pFile;

    pFile = fopen (name.c_str(), "w");

    fprintf (pFile, " (int)input->header.seq, \
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
             (float)ndt.getScore(),\
             (float)downSampleTime,\
             (float)alignTime,\
             (float)matchingTime,\
             (int)scan.size(), \
             (int)transformed_scan.size(),\
             (float)initial_guess_error,\
             (float)initial_guess_yaw_error,\
             (float)layout_non_ground_entropy,\
             (float)layout_all_entropy,\
             (float)x_points_feature_entropy,\
             (float)y_points_feature_entropy,\
             (float)yaw_ndt_normal_diff_entropy\n");

    fclose (pFile);


    name = save_path + "x_points_histogram.csv";
    pFile = fopen (name.c_str(), "w");
    fclose (pFile);

    name = save_path + "y_points_histogram.csv";
    pFile = fopen (name.c_str(), "w");
    fclose (pFile);


    name = save_path + "layout_histogram_nonground.csv";
    pFile = fopen (name.c_str(), "w");
    fclose (pFile);


    name = save_path + "layout_histogram_all.csv";
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
    ndt.setResolution(gridSize);
    ndt.setSearchResolution(gridSize);
    ndt.setStepSize(step_size);
    ndt.setTransformationEpsilon(trans_eps);


    // Setting point cloud to be aligned to.
    ndt.setInputTarget(map_ptr);
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



    ndt_map_pub = nh.advertise<visualization_msgs::MarkerArray>("visualization_marker_array", 10000);
    carTrajectory_pub = nh.advertise<visualization_msgs::Marker>("CarTrajectory3D_line", 1000);

    ros::Subscriber scan_sub = nh.subscribe("velodyne_points", 1000, scan_callback);


    publish_pointCloud(*map_ptr, map_pub, "map");




    ros::spin();



    return 0;
}

