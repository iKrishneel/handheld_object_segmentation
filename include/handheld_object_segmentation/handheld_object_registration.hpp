
#pragma once
#ifndef _HANDHELD_OBJECT_REGISTRATION_H_
#define _HANDHELD_OBJECT_REGISTRATION_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_listener.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PolygonStamped.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types_conversion.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/distances.h>
#include <pcl/common/centroid.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/registration/default_convergence_criteria.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class HandheldObjectRegistration {
   
 private:
    typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::PointCloud2, geometry_msgs::PolygonStamped,
    sensor_msgs::Image> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_point_;
    message_filters::Subscriber<geometry_msgs::PolygonStamped> sub_rect_;
    message_filters::Subscriber<sensor_msgs::Image> sub_mask_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::Normal NormalT;
    typedef pcl::PointXYZRGBNormal PointNormalT;
    typedef pcl::PointCloud<PointT> PointCloud;
    typedef pcl::PointCloud<NormalT> PointNormal;
    typedef pcl::PointCloud<PointNormalT> PointCloudNormal;
    typedef Eigen::Matrix<float, 4, 4> Matrix4f;
    typedef pcl::IterativeClosestPointWithNormals<
       PointNormalT, PointNormalT> ICP;

    bool update_model_;
    int iter_counter_;
    cv::Rect_<int> rect_;
    PointCloudNormal::Ptr model_points_;
    Matrix4f prev_trans_;
    cv::Size input_size_;
    int KERNEL_WSIZE_;

    ICP::Ptr icp_;
    pcl::VoxelGrid<PointNormalT> voxel_grid_;
    pcl::IntegralImageNormalEstimation<PointT, NormalT> ne_;

    pcl::search::KdTree<PointNormalT>::Ptr tree_;
    std::vector<pcl::PointIndices> cluster_indices_;
    pcl::EuclideanClusterExtraction<PointNormalT> ec_;

   
 protected:
    virtual void onInit();
    virtual void subscribe();
    virtual void unsubscribe();

    boost::mutex lock_;
    ros::NodeHandle pnh_;
    ros::Publisher pub_cloud_;

 public:
    HandheldObjectRegistration();
    void cloudCB(const sensor_msgs::PointCloud2::ConstPtr &,
                 const geometry_msgs::PolygonStamped::ConstPtr &,
                 const sensor_msgs::Image::ConstPtr &);
    void depthMapFromCloud(cv::Mat &, const PointCloud::Ptr,
                           const float = 10.0f);
    bool registrationICP(PointCloudNormal::Ptr, Matrix4f &,
                         const PointCloudNormal::Ptr,
                         const PointCloudNormal::Ptr);
    void fastSeedRegionGrowing(PointCloudNormal::Ptr,
                               const PointCloud::Ptr,
                               const PointNormal::Ptr);
    int seedVoxelConvexityCriteria(Eigen::Vector4f, Eigen::Vector4f,
                                   Eigen::Vector4f, Eigen::Vector4f,
                                   Eigen::Vector4f, const float = 0.0f);
    void getNormals(
       PointNormal::Ptr normals, const PointCloud::Ptr cloud);
   
    void seededRegionGrowingFromRect(PointCloudNormal::Ptr,
                                     const PointCloud::Ptr,
                                     const PointNormal::Ptr,
                                     const cv::Rect_<int>);
    void spatialClustering(PointCloudNormal::Ptr);
};


#endif /* _HANDHELD_OBJECT_REGISTRATION_H_ */
