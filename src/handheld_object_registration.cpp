
#include <handheld_object_segmentation/handheld_object_registration.hpp>

HandheldObjectRegistration::HandheldObjectRegistration() :
    iter_counter_(0), update_model_(true),
    KERNEL_WSIZE_(0) {

    this->tree_ = pcl::search::KdTree<PointNormalT>::Ptr(
       new pcl::search::KdTree<PointNormalT>);

    this->ec_.setClusterTolerance(0.015);
    this->ec_.setMinClusterSize(100);
    this->ec_.setMaxClusterSize(20000);
    this->ec_.setSearchMethod(this->tree_);
    
    this->model_points_ = PointCloudNormal::Ptr(new PointCloudNormal);
    this->icp_ = ICP::Ptr(new ICP);
    this->icp_->setMaximumIterations(5);
    this->icp_->setRANSACOutlierRejectionThreshold(0.06);
    this->icp_->setRANSACIterations(500);
    this->icp_->setTransformationEpsilon(1e-8);
    this->icp_->setUseReciprocalCorrespondences(true);
    this->icp_->setMaxCorrespondenceDistance(0.02);

    pcl::registration::TransformationEstimationPointToPlaneLLS<
       PointNormalT, PointNormalT>::Ptr trans_svd(
          new pcl::registration::TransformationEstimationPointToPlaneLLS<
          PointNormalT, PointNormalT>);
    this->icp_->setTransformationEstimation(trans_svd);

    const float leaf_size = 0.005f;
    this->voxel_grid_.setLeafSize(leaf_size, leaf_size, leaf_size);
    
    this->onInit();
}

void HandheldObjectRegistration::onInit() {
    this->subscribe();
    this->pub_cloud_ = this->pnh_.advertise<sensor_msgs::PointCloud2>(
       "/cloud", 1);
}

void HandheldObjectRegistration::subscribe() {
    this->sub_point_.subscribe(this->pnh_, "points", 1);
    this->sub_rect_.subscribe(this->pnh_, "rect", 1);
    this->sub_mask_.subscribe(this->pnh_, "mask", 1);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
       SyncPolicy> >(100);
    this->sync_->connectInput(this->sub_point_, this->sub_rect_,
                              this->sub_mask_);
    this->sync_->registerCallback(
       boost::bind(&HandheldObjectRegistration::cloudCB,
                   this, _1, _2, _3));
}

void HandheldObjectRegistration::unsubscribe() {
    this->sub_point_.unsubscribe();
    this->sub_rect_.unsubscribe();
}

void HandheldObjectRegistration::cloudCB(
    const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
    const geometry_msgs::PolygonStamped::ConstPtr &rect_msg,
    const sensor_msgs::Image::ConstPtr &mask_msg) {
   
    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    if (cloud->empty() || rect_msg->polygon.points.size() == 0) {
      ROS_ERROR("[::cloudCB]: EMPTY INPUTS");
      return;
    }

    cv_bridge::CvImagePtr cv_ptr;
    cv::Mat im_mask;
    try {
       cv_ptr = cv_bridge::toCvCopy(mask_msg,
                                    sensor_msgs::image_encodings::MONO8);
       im_mask = cv_ptr->image.clone();
    } catch (cv_bridge::Exception& e) {
       ROS_ERROR("cv_bridge exception: %s", e.what());
       return;
    }

    
    assert((cloud->width != 1 || cloud->height != 1)  &&
           "\033[31m UNORGANIZED INPUT CLOUD \033[0m");

    
    this->input_size_.width = cloud->width - (KERNEL_WSIZE_ * 2);
    this->input_size_.height = cloud->height - (KERNEL_WSIZE_ * 2);

    
    int x = rect_msg->polygon.points[0].x;
    int y = rect_msg->polygon.points[0].y;
    int width = rect_msg->polygon.points[1].x - x;
    int height = rect_msg->polygon.points[1].y - y;
    this->rect_ = cv::Rect_<int>(x, y, width, height);
    
    std::clock_t start;
    start = std::clock();
    
    PointNormal::Ptr normals(new PointNormal);
    this->getNormals(normals, cloud);

    //! get points
    PointCloudNormal::Ptr mask_cloud(new PointCloudNormal);

    //! mask out the region around the rect
    PointCloudNormal::Ptr src_pts(new PointCloudNormal);
    for (int j = y; j < y + height; j++) {
       for (int i = x; i < x + width; i++) {
          int index = i + j * cloud->width;
          PointNormalT pt;
          pt.x = cloud->points[index].x;
          pt.y = cloud->points[index].y;
          pt.z = cloud->points[index].z;
          pt.r = cloud->points[index].r;
          pt.g = cloud->points[index].g;
          pt.b = cloud->points[index].b;
          pt.normal_x = normals->points[index].normal_x;
          pt.normal_y = normals->points[index].normal_y;
          pt.normal_z = normals->points[index].normal_z;
          src_pts->push_back(pt);

          int val = static_cast<int>(im_mask.at<uchar>(j, i));
          if (val != 0) {
             mask_cloud->push_back(pt);
          }
       }
    }
    
    // this->voxel_grid_.setInputCloud(src_pts);
    // this->voxel_grid_.filter(*src_pts);

    this->voxel_grid_.setInputCloud(mask_cloud);
    this->voxel_grid_.filter(*mask_cloud);
    
    //! clustering
    // this->spatialClustering(src_pts);
    
    //! demean
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*src_pts, centroid);
    pcl::demeanPointCloud<PointNormalT, float>(*src_pts, centroid, *src_pts);
    
    if (this->iter_counter_++ == 0) {
       pcl::copyPointCloud<PointNormalT, PointNormalT>(
          *mask_cloud, *model_points_);
       this->prev_trans_ = Eigen::Matrix4f::Identity();
    } else {
       PointCloudNormal::Ptr align_points(new PointCloudNormal);
       Matrix4f transformation;
       bool is_ok = this->registrationICP(align_points, transformation,
                                          mask_cloud, mask_cloud);

       
       std::cout << transformation  << "\n";
       
       /*
       if (is_ok) {
          // src_pts->clear();
          // *src_pts = *align_points;

          bool pub_tf = false;
          if (pub_tf) {
             Eigen::Affine3f trans;
             trans.matrix() = transformation * prev_trans_;
             Eigen::Vector4f c;
             pcl::compute3DCentroid(*align_points, c);
             trans.translation().matrix() = Eigen::Vector3f(c[0], c[1], c[2]);

             tf::Transform tfTransformation;
             tf::transformEigenToTF((Eigen::Affine3d) trans,
                                    tfTransformation);

             static tf::TransformBroadcaster tfBroadcaster;
             tfBroadcaster.sendTransform(tf::StampedTransform(
                                            tfTransformation,
                                            cloud_msg->header.stamp,
                                            "/camera_rgb_optical_frame",
                                            "/pose"));
             prev_trans_ *= transformation;
          }
          
          if (this->update_model_) {
             this->model_points_->clear();
             pcl::copyPointCloud<PointNormalT, PointNormalT>(
                *src_pts, *model_points_);
          }
       }
       */
    }

    float duration = (std::clock() - start) /
       static_cast<float>(CLOCKS_PER_SEC);
    ROS_INFO("PROCESS TIME: %3.3f", duration);

    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*mask_cloud, ros_cloud);
    ros_cloud.header = cloud_msg->header;
    this->pub_cloud_.publish(ros_cloud);
}

void HandheldObjectRegistration::spatialClustering(
    PointCloudNormal::Ptr in_cloud) {
    if (in_cloud->empty()) {
       ROS_ERROR("[::spatialClustering]: EMPTY INPUT");
       in_cloud->clear();
       return;
    }
    PointCloudNormal::Ptr temp_cloud(new PointCloudNormal);
    
    this->voxel_grid_.setInputCloud(in_cloud);
    this->voxel_grid_.filter(*temp_cloud);
    
    this->tree_->setInputCloud(temp_cloud);
    cluster_indices_.clear();
    this->ec_.setInputCloud(temp_cloud);
    this->ec_.extract(cluster_indices_);
    int max_size = 0;
    int max_index = -1;
    for (auto it = cluster_indices_.begin();
         it != cluster_indices_.end(); it++) {
       if (max_size < it->indices.size()) {
          max_size = it->indices.size();
          max_index = std::distance(cluster_indices_.begin(), it);
       }
    }

    if (max_index == -1) {
       return;
    }
    
    Eigen::Vector3f min_point = Eigen::Vector3f(1000.0f, 1000.0f, 1000.0f);
    Eigen::Vector3f max_point = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < cluster_indices_[max_index].indices.size(); i++) {
       int index = cluster_indices_[max_index].indices[i];
       PointNormalT pt = temp_cloud->points[index];
       min_point(0) = pt.x < min_point(0) ? pt.x : min_point(0);
       min_point(1) = pt.y < min_point(1) ? pt.y : min_point(1);
       min_point(2) = pt.z < min_point(2) ? pt.z : min_point(2);
       max_point(0) = pt.x > max_point(0) ? pt.x : max_point(0);
       max_point(1) = pt.y > max_point(1) ? pt.y : max_point(1);
       max_point(2) = pt.z > max_point(2) ? pt.z : max_point(2);
    }

    temp_cloud->clear();
    for (int i = 0; i < in_cloud->size(); i++) {
       PointNormalT pt = in_cloud->points[i];
       if (pt.x > min_point(0) && pt.y > min_point(1) && pt.z > min_point(2) &&
           pt.x < max_point(0) && pt.y < max_point(1) && pt.z < max_point(2)) {
          temp_cloud->push_back(pt);
       }
    }
    in_cloud->clear();
    *in_cloud = *temp_cloud;
}

void HandheldObjectRegistration::seededRegionGrowingFromRect(
    PointCloudNormal::Ptr out_cloud, const PointCloud::Ptr cloud,
    const PointNormal::Ptr normals, const cv::Rect_<int> rect) {
    if (cloud->empty() || normals->empty() ||
        cloud->size() != normals->size()) {
       ROS_ERROR("[::seededRegionGrowingFromRect]: EMPTY INPUT");
       return;
    }
    
    this->fastSeedRegionGrowing(out_cloud, cloud, normals);
    this->voxel_grid_.setInputCloud(out_cloud);
    this->voxel_grid_.filter(*out_cloud);
}


void HandheldObjectRegistration::depthMapFromCloud(
    cv::Mat &im_depth, const PointCloud::Ptr in_cloud,
    const float max_dist) {
    im_depth = cv::Mat::zeros(in_cloud->height, in_cloud->width,
                              CV_32FC1);  //! move to global
    int index = -1;
    for (int j = 0; j < im_depth.rows; j++) {
       for (int i = 0; i < im_depth.cols; i++) {
          index = i + j * im_depth.cols;
          im_depth.at<float>(j, i) = in_cloud->points[index].z <
             max_dist ? in_cloud->points[index].z : 0.0f;
       }
    }
}

void HandheldObjectRegistration::getNormals(
    PointNormal::Ptr normals, const PointCloud::Ptr cloud) {
    if (cloud->empty()) {
       ROS_ERROR("-Input cloud is empty in normal estimation");
       return;
    }
    // pcl::IntegralImageNormalEstimation<PointT, NormalT> ne_;
    ne_.setNormalEstimationMethod(ne_.AVERAGE_3D_GRADIENT);
    ne_.setMaxDepthChangeFactor(0.02f);
    ne_.setNormalSmoothingSize(10.0f);
    ne_.setInputCloud(cloud);
    ne_.compute(*normals);
}

bool HandheldObjectRegistration::registrationICP(
    PointCloudNormal::Ptr align_points, Matrix4f &transformation,
    const PointCloudNormal::Ptr src_points,
    const PointCloudNormal::Ptr target_points) {
    if (src_points->empty() || target_points->empty()) {
       ROS_ERROR("- ICP FAILED. EMPTY INPUT");
       return false;
    }

    this->icp_->setInputSource(src_points);
    this->icp_->setInputTarget(target_points);

    ROS_WARN("SOLVING");
    
    this->icp_->align(*align_points);

    ROS_WARN("DONE");
    
    
    transformation = this->icp_->getFinalTransformation();
    
    // return (this->icp_->hasConverged());
    return false;
    
    
}

void HandheldObjectRegistration::fastSeedRegionGrowing(
    PointCloudNormal::Ptr src_points, const PointCloud::Ptr cloud,
    const PointNormal::Ptr normals) {
    if (cloud->empty() || normals->size() != cloud->size()) {
       return;
    }
    int seed_index = ((rect_.x + rect_.width/2) - KERNEL_WSIZE_)  +
       ((rect_.y + rect_.height/2) - KERNEL_WSIZE_) * input_size_.width;
    
    Eigen::Vector4f seed_point = cloud->points[seed_index].getVector4fMap();
    Eigen::Vector4f seed_normal = normals->points[
       seed_index].getNormalVector4fMap();

    std::vector<int> labels(static_cast<int>(cloud->size()), -1);

    const int window_size = 3;
    const int wsize = window_size * window_size;
    const int lenght = std::floor(window_size/2);

    std::vector<int> processing_list;
    for (int j = -lenght; j <= lenght; j++) {
       for (int i = -lenght; i <= lenght; i++) {
              int index = (seed_index + (j * input_size_.width)) + i;
          if (index >= 0 && index < cloud->size()) {
             processing_list.push_back(index);
          }
       }
    }
    std::vector<int> temp_list;
    while (true) {
       if (processing_list.empty()) {
          break;
       }
       temp_list.clear();
       for (int i = 0; i < processing_list.size(); i++) {
          int idx = processing_list[i];
          if (labels[idx] == -1) {
             Eigen::Vector4f c = cloud->points[idx].getVector4fMap();
             Eigen::Vector4f n = normals->points[idx].getNormalVector4fMap();
             if (this->seedVoxelConvexityCriteria(
                    seed_point, seed_normal, seed_point, c, n, -0.01) == 1) {
                labels[idx] = 1;

                for (int j = -lenght; j <= lenght; j++) {
                   for (int k = -lenght; k <= lenght; k++) {
                      int index = (idx + (j * input_size_.width)) + k;
                      if (index >= 0 && index < cloud->size()) {
                         temp_list.push_back(index);
                      }
                   }
                }
             }
          }
       }
       processing_list.clear();
       processing_list.insert(processing_list.end(), temp_list.begin(),
                              temp_list.end());
    }
    src_points->clear();
    for (int i = 0; i < labels.size(); i+=5) {
       if (labels[i] != -1) {
          PointNormalT pt;
          pt.x = cloud->points[i].x;
          pt.y = cloud->points[i].y;
          pt.z = cloud->points[i].z;
          pt.r = cloud->points[i].r;
          pt.g = cloud->points[i].g;
          pt.b = cloud->points[i].b;
          pt.normal_x = normals->points[i].normal_x;
          pt.normal_y = normals->points[i].normal_y;
          pt.normal_z = normals->points[i].normal_z;
          src_points->push_back(pt);
       }
    }
}

int HandheldObjectRegistration::seedVoxelConvexityCriteria(
    Eigen::Vector4f seed_point, Eigen::Vector4f seed_normal,
    Eigen::Vector4f c_centroid, Eigen::Vector4f n_centroid,
    Eigen::Vector4f n_normal, const float thresh) {
    float pt2seed_relation = FLT_MAX;
    float seed2pt_relation = FLT_MAX;
    pt2seed_relation = (n_centroid - seed_point).dot(n_normal);
    seed2pt_relation = (seed_point - n_centroid).dot(seed_normal);
    if (seed2pt_relation > thresh && pt2seed_relation > thresh) {
       return 1;
    } else {
       return -1;
    }
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "handheld_object_registration");
    HandheldObjectRegistration dgt;
    ros::spin();
    return 0;
}
