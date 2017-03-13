/*
 * =====================================================================================
 *
 *       Filename:  ProbabilityMapping.cc
 *
 *    Description:
 *
 *        Version:  0.1
 *        Created:  01/21/2016 10:39:12 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Josh Tang, Rebecca Frederick
 *
 *        version: 1.0
 *        created: 8/9/2016
 *        Log: fix a lot of bug, Almost rewrite the code.
 *
 *        author: He Yijia
 *
 *
 *       version: 1.5
 *       created: 02/23/2017
 *       Log: fix bug
 *
 *       author: Liu Xiaoyue
 *
 * =====================================================================================
 */

#include <cmath>
#include <opencv2/opencv.hpp>
#include <numeric>
#include "ProbabilityMapping.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "ORBmatcher.h"
#include "LocalMapping.h"
#include <stdint.h>
#include <stdio.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#define DEBUG 1
//#define InterKeyFrameChecking

void saveMatToCsv(cv::Mat data, std::string filename)
{
    std::ofstream outputFile(filename.c_str());
    outputFile << cv::format(data,"CSV")<<std::endl;
    outputFile.close();
}

template<typename T>
float bilinear(const cv::Mat& img, const float& y, const float& x)
{
    int x0 = (int)std::floor(x);
    int y0 = (int )std::floor(y);
    int x1 = x0 + 1;
    int y1 =  y0 + 1;

    float x0_weight = x1 - x;
    float y0_weight = y1 - y;
    float x1_weight = 1.0f - x0_weight;
    float y1_weight = 1.0f - y0_weight;
/*
    if(x1 >= img.cols || y1 >= img.rows)
    {
        return 1000;   // return a large error
    }
*/
  //  std::cout<<"image: "<<(float) img.at<T>(y0 , x0 )<<"  "<<(float) img.at<T>(y1 , x1 )<<"  "<< (float)img.at<T>(y1 , x0 )<<"  "<<(float) img.at<T>(y0 , x1 )<<std::endl;
    float interpolated =
            img.at<T>(y0 , x0 ) * x0_weight + img.at<T>(y0 , x1)* x1_weight +
            img.at<T>(y1 , x0 ) * x0_weight + img.at<T>(y1 , x1)* x1_weight +
            img.at<T>(y0 , x0 ) * y0_weight + img.at<T>(y1 , x0)* y1_weight +
            img.at<T>(y0 , x1 ) * y0_weight + img.at<T>(y1 , x1)* y1_weight ;

  return (interpolated * 0.25f);
}

ProbabilityMapping::ProbabilityMapping(ORB_SLAM2::Map* pMap):mpMap(pMap)
{
 mbFinishRequested = false; //init
}

void ProbabilityMapping::Run()
{
    while(1)
    {
        if(CheckFinish()) break;
        sleep(1);
        SemiDenseLoop();
        
    }
}

void ProbabilityMapping::SemiDenseLoop(){

  unique_lock<mutex> lock(mMutexSemiDense);

  vector<ORB_SLAM2::KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
  cout<<"semidense_Info:    vpKFs.size()--> "<<vpKFs.size()<<std::endl;
  if(vpKFs.size() < covisN+3){return;}

  for(size_t i =0;i < vpKFs.size(); i++ )
  {

      ORB_SLAM2::KeyFrame* kf = vpKFs[i];
      if(kf->isBad() || kf->semidense_flag_)
        continue;

        std::vector<ORB_SLAM2::KeyFrame*> closestMatches = kf->GetBestCovisibilityKeyFrames(covisN);
        if(closestMatches.size() < covisN) {continue;}

        float max_inverse_depth;
        float min_inverse_depth;
        // get max_dephth  and min_depth in current key frame to limit search range
        StereoSearchConstraints(kf, &min_inverse_depth, &max_inverse_depth);

        cv::Mat image = kf->GetImage();

        std::vector <cv::Mat> F;
        F.clear();
        for(size_t j=0; j<closestMatches.size(); j++)
        {
          ORB_SLAM2::KeyFrame* kf2 = closestMatches[ j ];
          cv::Mat F12 = ComputeFundamental(kf,kf2);
          F.push_back(F12);
        }


        for(int y = 0+2; y < image.rows-2; y++)
        {
          for(int x = 0+2; x< image.cols-2; x++)
          {

            if(kf->GradImg.at<float>(y,x) < lambdaG){continue;}
            float pixel =(float) image.at<uchar>(y,x); //maybe it should be cv::Mat

            std::vector<depthHo> depth_ho;
            depth_ho.clear();
            for(size_t j=0; j<closestMatches.size(); j++)
            {
                ORB_SLAM2::KeyFrame* kf2 = closestMatches[ j ];
                cv::Mat F12 = F[j];

                float best_u(0.0),best_v(0.0);
                depthHo dh;

                EpipolarSearch(kf, kf2, x, y, pixel, min_inverse_depth, max_inverse_depth, &dh,F12,best_u,best_v,kf->GradTheta.at<float>(y,x));

                if (dh.supported && 1/dh.invdepth > 0.0)
                {
                      depth_ho.push_back(dh);
                  }
            }

            if (depth_ho.size() >= lambdaN) {
                depthHo dh_temp;
                InverseDepthHypothesisFusion(depth_ho, dh_temp);
                if(dh_temp.supported)
                {
                  kf->depth_map_.at<float>(y,x) = dh_temp.invdepth;   //  used to do IntraKeyFrameDepthChecking
                  kf->depth_sigma_.at<float>(y,x) = dh_temp.sigma;
                }
            }

          }
        }
      
         IntraKeyFrameDepthChecking( kf->depth_map_,  kf->depth_sigma_, kf->GradImg);

}
//#ifdef InterKeyFrameChecking
  for(size_t i =0;i < vpKFs.size(); i++ )
  {
     ORB_SLAM2::KeyFrame* kf = vpKFs[i];

     if(kf->isBad() || kf->semidense_flag_)continue;
       
      ofstream f;
        
      char filename[25];
      sprintf(filename,"%lf.txt",kf->mTimeStamp); 

      f.open(filename);

      InterKeyFrameDepthChecking(kf);

      for(int y = 0+2; y < kf->im_.rows-2; y++)
      {
        for(int x = 0+2; x< kf->im_.cols-2; x++)
        {

          if(kf->depth_map_.at<float>(y,x) < 0.0001) {
            f<<0.0<<" "<<0.0<<" "<<0.0<<endl;
            continue;
          }

            float inv_d = kf->depth_map_.at<float>(y,x);
            float Z = 1/inv_d ;
            float X = Z *(x- kf->cx ) / kf->fx;
            float Y = Z*(y- kf->cy ) / kf->fy;

            cv::Mat Pc = (cv::Mat_<float>(4,1) << X, Y , Z, 1); // point in camera frame.
            cv::Mat Twc = kf->GetPoseInverse();
            cv::Mat pos = Twc * Pc;

            kf->SemiDensePointSets_.at<float>(y,3*x+0) = pos.at<float>(0);
            kf->SemiDensePointSets_.at<float>(y,3*x+1) = pos.at<float>(1);
            kf->SemiDensePointSets_.at<float>(y,3*x+2) = pos.at<float>(2);

            f<<kf->SemiDensePointSets_.at<float>(y,3*x+0)<<" "<<kf->SemiDensePointSets_.at<float>(y,3*x+1)<<" "<<kf->SemiDensePointSets_.at<float>(y,3*x+2)<<endl;
        }
      }

      kf->semidense_flag_ = true;
 
      f.close();
      //cout<<endl<<"SaveSemiDenseMappoints saved!"<<endl;
}
//#endif
      cout<<"semidense_Info:    vpKFs.size()--> "<<vpKFs.size()<<std::endl;
}

void ProbabilityMapping::StereoSearchConstraints(ORB_SLAM2::KeyFrame* kf, float* min_inverse_depth, float* max_inverse_depth){
  std::vector<float> orb_inverse_depths = kf->GetAllPointDepths();

  float sum = std::accumulate(orb_inverse_depths.begin(), orb_inverse_depths.end(), 0.0);
  float mean = sum / orb_inverse_depths.size();

  std::vector<float> diff(orb_inverse_depths.size());
  std::transform(orb_inverse_depths.begin(), orb_inverse_depths.end(), diff.begin(), std::bind2nd(std::minus<float>(), mean));
  float variance = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0)/orb_inverse_depths.size();
  float stdev = std::sqrt(variance);

  *max_inverse_depth = mean + 2 * stdev;
  *min_inverse_depth = mean - 2 * stdev;
}

void ProbabilityMapping::EpipolarSearch(ORB_SLAM2::KeyFrame* kf1, ORB_SLAM2::KeyFrame *kf2, const int x, const int y, float pixel,
    float min_inverse_depth, float max_inverse_depth, depthHo *dh,cv::Mat F12,float& best_u,float& best_v,float th_pi)
{

  float a = x*F12.at<float>(0,0)+y*F12.at<float>(1,0)+F12.at<float>(2,0);
  float b = x*F12.at<float>(0,1)+y*F12.at<float>(1,1)+F12.at<float>(2,1);
  float c = x*F12.at<float>(0,2)+y*F12.at<float>(1,2)+F12.at<float>(2,2);

  if((a/b)< -4 || a/b> 4) return;   // if epipolar direction is approximate to perpendicular, we discard it.  May be product wrong match.

  float old_err = 1000.0;
  float best_photometric_err = 0.0;
  float best_gradient_modulo_err = 0.0;
  float best_pixel = 0.0;

  int vj, uj_plus, vj_plus, uj_minus, vj_minus;
  float g, q, denomiator, ustar, ustar_var;

  float umin(0.0),umax(0.0);
  GetSearchRange(umin,umax,x,y,min_inverse_depth,max_inverse_depth,kf1,kf2);
  //for(int uj = 0; uj < image.cols; uj++)// FIXME should use  min and max depth
  for(int uj = std::floor(umin); uj < std::ceil(umax)+1; uj++)// FIXME should use  min and max depth
  {
    vj =-(int)( (a/b)*uj+(c/b));
    if(vj<0 || vj > kf2->im_.rows ){continue;}

    // condition 1:
    if( kf2->GradImg.at<float>(vj,uj) < lambdaG){continue;}

    // condition 2:
    float th_epipolar_line = cv::fastAtan2(-a/b,1);
    float temp_gradth =  kf2->GradTheta.at<float>(vj,uj) ;
    if( temp_gradth > 270) temp_gradth =  temp_gradth - 360;
    if( temp_gradth > 90 &&  temp_gradth<=270)
      temp_gradth =  temp_gradth - 180;
    if(th_epipolar_line>270) th_epipolar_line = th_epipolar_line - 360;
    if(abs(abs(temp_gradth - th_epipolar_line) - 90)< 10 ){continue;}

    // condition 3:
    //if(abs(th_grad - ( th_pi + th_rot )) > lambdaTheta)continue;
    /*const vector<cv::KeyPoint> &vKeysUn1 = pk1->mvKeysUn;
    const vector<cv::KeyPoint> &vKeysUn2 = pk2->mvKeysUn;
    
    float rot=0.0;
    vector<cv::KeyPoint> mvKeys1= kf2.mvKeysUn;
    vector<cv::KeyPoint> mvKeys2= kf1.mvKeysUn;

    for(int i=0;i<5;i++){
      size_t n=rand()%kf1.mvKeysUn.size();      
      rot +=(kf1.mvKeysUn[n].angle - kf2.mvKeysUn[n].angle);

    }
    rot = rot/10;*/
    
    if(abs( kf2->GradTheta.at<float>(vj,uj) -  th_pi - 3 ) > lambdaTheta)continue;

     float photometric_err = pixel - bilinear<uchar>(kf2->im_,-((a/b)*uj+(c/b)),uj);
     float gradient_modulo_err = kf1->GradImg.at<float>(y,x)  - bilinear<float>( kf2->GradImg,-((a/b)*uj+(c/b)),uj);

   // std::cout<<bilinear<float>(grad2mag,-((a/b)*uj+(c/b)),uj)<<"  grad : "<< grad2mag.at<float>(vj,uj)<<std::endl;
   // std::cout<<bilinear<uchar>(image,-((a/b)*uj+(c/b)),uj)<<"  image : "<< (float)image.at<uchar>(vj,uj)<<std::endl;

   //float err = (photometric_err*photometric_err + (gradient_modulo_err*gradient_modulo_err)/THETA)/(image_stddev.at<double>(0,0)*image_stddev.at<double>(0,0));
    float err = ((photometric_err*photometric_err  + (gradient_modulo_err*gradient_modulo_err)/THETA)/(kf2->I_stddev*kf2->I_stddev));
    if(err < old_err)
    {
      best_pixel = uj;
      old_err = err;
      best_photometric_err = photometric_err;
      best_gradient_modulo_err = gradient_modulo_err;
    }
  }

  if(old_err < 400.0)
  {

     uj_plus = best_pixel + 1;
     vj_plus = -((a/b)*uj_plus + (c/b));
     uj_minus = best_pixel - 1;
     vj_minus = -((a/b)*uj_minus + (c/b));

     g = ((float)kf2->im_.at<uchar>(vj_plus, uj_plus) -(float) kf2->im_.at<uchar>(vj_minus, uj_minus))/2.0;
     q = ( kf2->GradImg.at<float>(vj_plus, uj_plus) -  kf2->GradImg.at<float>(vj_minus, uj_minus))/2.0;
/*
     if(vj_plus< 0)   //  if abs(a/b) is large,   a little step for uj, may produce a large change on vj. so there is a bug !!!  vj_plus may <0
     {
       std::cout<<"vj_plus: "<<vj_plus<<" a/b: "<<a/b<<" c/b: "<<c/b<<std::endl;
       std::cout<<"best_pixel: "<<best_pixel<<" vj: "<<vj<<" old_err "<<old_err<<std::endl;
       std::cout<<"uj_plus: "<<uj_plus<<std::endl;
     }
*/

  //   g = (bilinear<uchar>(image,-((a/b)*uj_plus+(c/b)),uj_plus) - bilinear<uchar>(image,-((a/b)*uj_minus+(c/b)),uj_minus))/2.0;
  //   g = (bilinear<float>(grad2mag,-((a/b)*uj_plus+(c/b)),uj_plus) - bilinear<float>(grad2mag,-((a/b)*uj_minus+(c/b)),uj_minus))/2.0;

     denomiator = (g*g + (1/THETA)*q*q);
     ustar = best_pixel + (g*best_photometric_err + (1/THETA)*q*best_gradient_modulo_err)/denomiator;
     ustar_var = (2*kf2->I_stddev*kf2->I_stddev/denomiator);
     //std::cout<< "g:"<<g<< "  q:"<<q<<"  I_err:"<<best_photometric_err<<"  g_err"<<best_gradient_modulo_err<< "   denomiator:"<<denomiator<< "  ustar:"<<ustar<<std::endl;

     //if(ustar_var > 5) return;

     best_u = ustar;
     best_v =  -( (a/b)*best_u + (c/b) );

    // GetPixelDepth(best_u, best_v, x ,y,kf1, kf2,dh->depth,dh);
    // GetPixelDepth(best_u, x , y, kf1, kf2, dh->depth);
    // dh->supported = true;
     ComputeInvDepthHypothesis(kf1, kf2, ustar, ustar_var, a, b, c, dh,x,y);
  }

}

void ProbabilityMapping::GetSearchRange(float& umin, float& umax, int px, int py,float mind,float maxd,
                                        ORB_SLAM2::KeyFrame* kf,ORB_SLAM2::KeyFrame* kf2)
{
  float fx = kf->fx;
  float cx = kf->cx;
  float fy = kf->fy;
  float cy = kf->cy;

  cv::Mat Rcw1 = kf->GetRotation();
  cv::Mat tcw1 = kf->GetTranslation();
  cv::Mat Rcw2 = kf2->GetRotation();
  cv::Mat tcw2 = kf2->GetTranslation();

  cv::Mat R21 = Rcw2*Rcw1.t();
  cv::Mat t21 = -Rcw2*Rcw1.t()*tcw1+tcw2;

  cv::Mat xp1=(cv::Mat_<float>(3,1) << (px-cx)/fx, (py-cy)/fy,1.0);  // inverse project.    if has distortion, this code shoud fix

  if(mind<0) mind = 0;
  cv::Mat xp2_min = R21*xp1/mind+t21;
  cv::Mat xp2_max = R21*xp1/maxd+t21;

  umin = fx*xp2_min.at<float>(0)/xp2_min.at<float>(2) + cx;
  umax = fx*xp2_max.at<float>(0)/xp2_max.at<float>(2) + cx;

  if(umin<0.0) umin = 0.0;
  if(umax<0.0) umax = 0.0;
  if(umin>(float)kf->im_.cols ) umin = (float)kf->im_.cols;
  if(umax>(float)kf->im_.cols)  umax = (float)kf->im_.cols;
}

void ProbabilityMapping::ComputeInvDepthHypothesis(ORB_SLAM2::KeyFrame* kf, ORB_SLAM2::KeyFrame* kf2, float ustar, float ustar_var,
                                                   float a, float b, float c,ProbabilityMapping::depthHo *dh, int x,int y)
{

  //float v_star=- ((a/b) * ustar + (c/b));
  float inv_pixel_depth =  0.0;

  // equation 8 comput depth
  GetPixelDepth(ustar, x , y,kf, kf2,inv_pixel_depth);
  // linear triangulation method
  // GetPixelDepth(ustar, pixel_y, x ,y,kf, kf2,inv_pixel_depth,dh);

  float ustar_min = ustar - sqrt(ustar_var);
  //int vstar_min = -((a/b)*ustar_min + (c/b));

  float inv_depth_min = 0.0;
  GetPixelDepth(ustar_min,x,y,kf,kf2, inv_depth_min);
  //(inv_frame_rot[2]*corrected_image.at<float>(ustarcx_min ,vstarcx_min)-fx*inv_frame_rot[0]*corrected_image.at<float>(ujcx,vjcx))/(-transform_data[2][ustarcx_min][vstarcx_min]+fx*transform_data[0]);

  float ustar_max = ustar +  sqrt(ustar_var);
  //int vstar_max = -((a/b)*ustar_max + (c/b));

  float inv_depth_max = 0.0;
  GetPixelDepth(ustar_max,x,y,kf, kf2,inv_depth_max);
  //(inv_frame_rot[2]*corrected_image.at<float>(ustarcx_max ,vstarcx_max)-fx*inv_frame_rot[0]*corrected_image.at<float>(ujcx,vjcx)/)/(-transform_data[2][ustarcx_max][vstarcx_max]+fx*transform_data[0]);

  // Equation 9
  float sigma_depth = cv::max(abs(inv_depth_max-inv_pixel_depth), abs(inv_depth_min-inv_pixel_depth));

  dh->invdepth = inv_pixel_depth;
  dh->sigma = sigma_depth;
  dh->supported = true;

}


void ProbabilityMapping::InverseDepthHypothesisFusion(const std::vector<depthHo>& h, depthHo& dist) {
    dist.invdepth = 0.0;
    dist.sigma = 0.0;
    dist.supported = false;

    std::vector<depthHo> compatible_ho;
    std::vector<depthHo> compatible_ho_temp;
    float chi = 0.0;

    for (size_t a=0; a < h.size(); a++) {

        compatible_ho_temp.clear();
        for (size_t b=0; b < h.size(); b++)
        {
          if (ChiTest(h[a], h[b], &chi))
            {compatible_ho_temp.push_back(h[b]);}// test if the hypotheses a and b are compatible
        }

        // test if hypothesis 'a' has the required support
        if (compatible_ho_temp.size() >= lambdaN )
        {
            compatible_ho.push_back(h[a]);
        }
    }

    // calculate the parameters of the inverse depth distribution by fusing hypotheses
    if (compatible_ho.size() >= lambdaN) {
        GetFusion(compatible_ho, dist, &chi);
    }
}


void ProbabilityMapping::IntraKeyFrameDepthChecking(cv::Mat& depth_map, cv::Mat& depth_sigma,const cv::Mat gradimg)
{
   //std::vector<std::vector<depthHo> > ho_new (depth_map.rows, std::vector<depthHo>(depth_map.cols, depthHo()) );
   cv::Mat depth_map_new = depth_map.clone();
   cv::Mat depth_sigma_new = depth_sigma.clone();

   int grow_cnt(0);
   for (int py = 2; py < (depth_map.rows - 2); py++)
   {
       for (int px = 2; px < (depth_map.cols - 2); px++)
       {

           if (depth_map.at<float>(py,px) < 0.0001)  // if  d ==0.0 : grow the reconstruction getting more density
           {

                  if(gradimg.at<float>(py,px)<lambdaG) continue;
                  //search supported  by at least 2 of its 8 neighbours pixels
                  std::vector< std::pair<float,float> > max_supported;

                  for( int  y = py - 1 ; y <= py+1; y++)
                    for( int  x = px - 1 ; x <= px+1; x++)
                    {

                      std::vector< std::pair<float,float> > supported;
                      if(x == px && y == py) continue;
                      for (int nx = px - 1; nx <= px + 1; nx++)
                          for (int ny = py - 1; ny <= py + 1; ny++)
                          {
                            if((x == nx && y == ny) || (nx == px && ny == py))continue;
                            if(ChiTest(depth_map.at<float>(y,x),depth_map.at<float>(ny,nx),depth_sigma.at<float>(y,x),depth_sigma.at<float>(ny,nx)))
                            {
                                     std::pair<float, float> depth;
                                     depth.first = depth_map.at<float>(ny,nx);
                                     depth.second = depth_sigma.at<float>(ny,nx);
                                     supported.push_back(depth);
                            }
                          }

                      if(supported.size()>0)   //  select the max supported neighbors
                      {
                        std::pair<float, float> depth;
                        depth.first = depth_map.at<float>(y,x);
                        depth.second = depth_sigma.at<float>(y,x);
                        //supported.push_back(depth);   // push (y,x) itself
                        max_supported.push_back(depth);

                      }
                    }

                  if(max_supported.size() > 1)
                  {
                    grow_cnt ++;
                    float d(0.0),s(0.0);
                    GetFusion(max_supported,d,s);
                    depth_map_new.at<float>(py,px) = d;
                    depth_sigma_new.at<float>(py,px) = s;

                  }

           }
           else
           {
             std::vector<depthHo> compatible_neighbor_ho;

             depthHo dha,dhb;
             dha.invdepth = depth_map.at<float>(py,px);
             dha.sigma = depth_sigma.at<float>(py,px);

             for (int y = py - 1; y <= py + 1; y++)
             {
                 for (int x = px - 1; x <= px + 1; x++)
                 {

                   if (x == px && y == py) continue;
                   if( depth_map.at<float>(y,x)> 0)
                   {
                     if(ChiTest(depth_map.at<float>(y,x),depth_map.at<float>(py,px),depth_sigma.at<float>(y,x),depth_sigma.at<float>(py,px)))
                     {
                       dhb.invdepth = depth_map.at<float>(y,x);
                       dhb.sigma = depth_sigma.at<float>(y,x);
                       compatible_neighbor_ho.push_back(dhb);
                     }

                   }
                 }
             }
             //compatible_neighbor_ho.push_back(dha);  // dont forget itself.

             if (compatible_neighbor_ho.size() > 2)
             {
                 depthHo fusion;
                 float min_sigma = 0.0;
                 GetFusion(compatible_neighbor_ho, fusion, &min_sigma);

                 depth_map_new.at<float>(py,px) = fusion.invdepth;
                 depth_sigma_new.at<float>(py,px) = min_sigma;

                 //ho_new[py][px].depth = fusion.depth;
                 //ho_new[py][px].sigma = min_sigma;
                // ho_new[py][px].supported = true;

             } else
             {
                 //ho_new[py][px].supported = false;   // outlier
                 depth_map_new.at<float>(py,px) = 0.0;
                 depth_sigma_new.at<float>(py,px) = 0.0;

             }

           }
       }
   }
  // ho = ho_new;
   //std::cout<<"intra key frame grow pixel number: "<<grow_cnt<<std::endl;
   depth_map = depth_map_new.clone();
   depth_sigma = depth_sigma_new.clone();

}

void ProbabilityMapping::InterKeyFrameDepthChecking(ORB_SLAM2::KeyFrame* currentKf)
{

    std::vector<ORB_SLAM2::KeyFrame*> neighbors;

    // option1: could just be the best covisibility keyframes
    neighbors = currentKf->GetBestCovisibilityKeyFrames(covisN);
    if(neighbors.size() < covisN) {return;}

    // for each pixel of keyframe_i, project it onto each neighbor keyframe keyframe_j
    // and propagate inverse depth

    std::vector <cv::Mat> Rji,tji;
    for(size_t j=0; j<neighbors.size(); j++)
    {
      ORB_SLAM2::KeyFrame* kf2 = neighbors[ j ];

      cv::Mat Rcw1 = currentKf->GetRotation();
      cv::Mat tcw1 = currentKf->GetTranslation();
      cv::Mat Rcw2 = kf2->GetRotation();
      cv::Mat tcw2 = kf2->GetTranslation();

      cv::Mat R21 = Rcw2*Rcw1.t();
      cv::Mat t21 = -Rcw2*Rcw1.t()*tcw1+tcw2;

      Rji.push_back(R21);
      tji.push_back(t21);

    }

    int cols = currentKf->im_.cols;
    int rows = currentKf->im_.rows;
    float fx = currentKf->fx;
    float fy = currentKf->fy;
    float cx = currentKf->cx;
    float cy = currentKf->cy;
    int remove_cnt(0);

    for (int py = 2; py <rows-2; py++) {
        for (int px = 2; px < cols-2; px++)  {

            if (currentKf->depth_map_.at<float>(py,px) < 0.0001) continue;   //  if d == 0.0  continue;

            float depthp = currentKf->depth_map_.at<float>(py,px);
            // count of neighboring keyframes in which there is at least one compatible pixel
            int compatible_neighbor_keyframes_count = 0;

            std::vector<compatible_pixels_> compatible_neighbor_keyframes_pixels_;

            cv::Mat xp=(cv::Mat_<float>(3,1) << (px-cx)/fx, (py-cy)/fy,1.0);// inverse project.    if has distortion, this code shoud fix

            for(size_t j=0; j<neighbors.size(); j++) {

                ORB_SLAM2::KeyFrame* pKFj = neighbors[j];
                cv::Mat K = pKFj->GetCalibrationMatrix();

 
                cv::Mat temp = Rji[j] * xp /depthp + tji[j];
                cv::Mat Xj = K*temp;
                Xj = Xj/Xj.at<float>(2);   //   u = u'/z   ,  v = v'/z

                // Eq (12)
                // compute the projection matrix to map 3D point from original image to 2D point in neighbor keyframe
                temp = Rji[j].row(2) * xp;
                float denom1 = temp.at<float>(0,0);
                temp = depthp * tji[j].at<float>(2);
                float denom2 = temp.at<float>(0,0);
                float depthj = depthp / (denom1 + denom2);

                float xj = Xj.at<float>(0);
                float yj = Xj.at<float>(1);

                std::vector<compatible_pixels_> compatible_pixels_temp;


                // look in 4-neighborhood pixel p_j,n around xj for compatible inverse depth

                if(xj < 0 || xj > cols || yj<0 || yj>rows) continue;

                int pxn = (int)std::floor(xj);
                int pyn = (int)std::floor(yj);


                for (int nx = pxn-1; nx <= pxn + 1; nx++) {
                    for (int ny = pyn-1; ny < pyn + 1; ny++) {
                        if ((nx == ny) || ((nx - pxn) && (ny - pyn))) continue;

                         compatible_pixels_ pixels_temp;

                        // Eq (13)
                        float depthjn = pKFj->depth_map_.at<float>(ny,nx);
                        float sigmajn = pKFj->depth_sigma_.at<float>(ny,nx);

                        float test = pow((depthj - depthjn), 2) / pow(sigmajn, 2);
                        if (test < 3.84) {

                            pixels_temp.pj = j;
                            pixels_temp.xj = nx;
                            pixels_temp.yj = ny;
                            
                            compatible_pixels_temp.push_back(pixels_temp);

                        }
                    }
                }

                //compatible_pixels_by_frame[j] = compatible_pixels; // is this a memory leak?
                // n_compatible_pixels += compatible_pixels.size();

                // at least one compatible pixel p_j,n must be found in at least lambdaN neighbor keyframes
                if (compatible_pixels_temp.size()) {
                  compatible_neighbor_keyframes_count++;
                  
                  for(int x=0;x<compatible_pixels_temp.size();x++){
                    compatible_neighbor_keyframes_pixels_.push_back(compatible_pixels_temp[x]);
                  
                  }
                }

            } // for j = 0...neighbors.size()-1

            // don't retain the inverse depth distribution of this pixel if not enough support in neighbor keyframes
            if (compatible_neighbor_keyframes_count < lambdaN )
            {
                currentKf->depth_map_.at<float>(py,px) = 0.0;
                remove_cnt++;

            }else{

              float depth_temp_;
              float min_temp=1000.0;

              for(int i = 0 ;i <compatible_neighbor_keyframes_pixels_.size();i++){
                size_t p = compatible_neighbor_keyframes_pixels_[i].pj;
                int xd = compatible_neighbor_keyframes_pixels_[i].xj;
                int yd = compatible_neighbor_keyframes_pixels_[i].yj;

                ORB_SLAM2::KeyFrame* neighbors_kf = neighbors[p];
                float invdp = neighbors_kf->depth_map_.at<float>(yd,xd);

                float temp = 0.0;

                for(int j = 0;j < compatible_neighbor_keyframes_pixels_.size();j++){
                  float res = 0.0;
                  size_t pjn = compatible_neighbor_keyframes_pixels_[j].pj;
                  int xjn = compatible_neighbor_keyframes_pixels_[j].xj;
                  int yjn = compatible_neighbor_keyframes_pixels_[j].yj;

                  ORB_SLAM2::KeyFrame* neighbors_kf2 = neighbors[pjn];
                  float invdjn = neighbors_kf2->depth_map_.at<float>(yjn,xjn);
                  float sigmajn = neighbors_kf2->depth_sigma_.at<float>(yjn,xjn);
                  cv::Mat rzji = Rji[pjn].row(2);
                  cv::Mat tzji = tji[pjn];

                  Equation14(invdjn,invdp,sigmajn,xp,rzji,tzji,&res);
                  temp += res;
                }

                if(temp < min_temp){
                  min_temp = temp;
                  depth_temp_ = invdp;
                }
              }
              currentKf->depth_map_.at<float>(py,px) = depth_temp_;
            }

        } // for py = 0...im.cols-1
    } // for px = 0...im.rows-1

    std::cout<<"Inter Key Frame checking , remove outlier: "<<remove_cnt<<std::endl;
}

// Equation (8)
void ProbabilityMapping::GetPixelDepth(float uj, int px, int py, ORB_SLAM2::KeyFrame* kf,ORB_SLAM2::KeyFrame* kf2, float &p) {

    float fx = kf->fx;
    float cx = kf->cx;
    float fy = kf->fy;
    float cy = kf->cy;

    float ucx = uj - cx;

    cv::Mat Rcw1 = kf->GetRotation();
    cv::Mat tcw1 = kf->GetTranslation();
    cv::Mat Rcw2 = kf2->GetRotation();
    cv::Mat tcw2 = kf2->GetTranslation();

    cv::Mat R21 = Rcw2*Rcw1.t();
    cv::Mat t21 = -Rcw2*Rcw1.t()*tcw1+tcw2;

    cv::Mat xp=(cv::Mat_<float>(3,1) << (px-cx)/fx, (py-cy)/fy,1.0);// inverse project.    if has distortion, this code shoud fix

    //GetXp(kf->GetCalibrationMatrix(), px, py, &xp);

    cv::Mat temp = R21.row(2) * xp * ucx;
    float num1 = temp.at<float>(0,0);
    temp = fx * (R21.row(0) * xp);
    float num2 = temp.at<float>(0,0);
    float denom1 = -t21.at<float>(2) * ucx;
    float denom2 = fx * t21.at<float>(0);

    p = (num1 - num2) / (denom1 + denom2);

}


void ProbabilityMapping::GetFusion(const std::vector<depthHo>& compatible_ho, depthHo& hypothesis, float* min_sigma) {
    hypothesis.invdepth = 0.0;
    hypothesis.sigma = 0.0;

    float temp_min_sigma = 100.0;
    float pjsj =0.0; // numerator
    float rsj =0.0; // denominator

    for (size_t j = 0; j < compatible_ho.size(); j++) {
        pjsj += compatible_ho[j].invdepth / pow(compatible_ho[j].sigma, 2);
        rsj += 1 / pow(compatible_ho[j].sigma, 2);
        if (pow(compatible_ho[j].sigma, 2) < pow(temp_min_sigma, 2)) {
            temp_min_sigma = compatible_ho[j].sigma;
        }
    }

    hypothesis.invdepth = pjsj / rsj;
    hypothesis.sigma = sqrt(1 / rsj);
    hypothesis.supported = true;

    if (min_sigma) {
        *min_sigma = temp_min_sigma;
    }
}

void ProbabilityMapping::GetFusion(const std::vector<std::pair <float,float> > supported, float& depth, float& sigma)
{
    int t = supported.size();
    float pjsj =0.0; // numerator
    float rsj =0.0; // denominator
    for(size_t i = 0; i< t; i++)
    {
      pjsj += supported[i].first / pow(supported[i].second, 2);
      rsj += 1 / pow(supported[i].second, 2);
    }

    depth = pjsj / rsj;
    sigma = sqrt(1 / rsj);
}

void ProbabilityMapping::Equation14(float invdjn, float invdp, float sigmajn, cv::Mat xp, cv::Mat rji, cv::Mat tji,float* res) {
    cv::Mat tempm = rji * xp;
    float tempf = tempm.at<float>(0,0);
    float djn = 1 / invdjn;
    float dp = 1 / invdp;
    
    *res = pow((djn - (dp * tempf) - tji.at<float>(2)) / (pow(djn, 2) * sigmajn), 2);
}

bool ProbabilityMapping::ChiTest(const depthHo& ha, const depthHo& hb, float* chi_val) {
    float num = (ha.invdepth - hb.invdepth)*(ha.invdepth - hb.invdepth);
    float chi_test = num / (ha.sigma*ha.sigma) + num / (hb.sigma*hb.sigma);
    if (chi_val)
        *chi_val = chi_test;
    return (chi_test < 5.99);  // 5.99 -> 95%
}

bool ProbabilityMapping::ChiTest(const float& a, const float& b, const float sigma_a,float sigma_b) {
    float num = (a - b)*(a - b);
    float chi_test = num / (sigma_a*sigma_a) + num / (sigma_b*sigma_b);
    return (chi_test < 5.99);  // 5.99 -> 95%
}

cv::Mat ProbabilityMapping::ComputeFundamental( ORB_SLAM2::KeyFrame *&pKF1,  ORB_SLAM2::KeyFrame *&pKF2) {
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = GetSkewSymmetricMatrix(t12);

    cv::Mat K1 = pKF1->GetCalibrationMatrix();
    cv::Mat K2 = pKF2->GetCalibrationMatrix();

    return K1.t().inv()*t12x*R12*K2.inv();
}

cv::Mat ProbabilityMapping::GetSkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
                                  v.at<float>(2),               0,-v.at<float>(0),
                                 -v.at<float>(1),  v.at<float>(0),              0);
}
