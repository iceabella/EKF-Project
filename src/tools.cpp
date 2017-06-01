#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
      * Calculate the RMSE
  */
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  //  * the estimation vector size should not be zero
  if(estimations.size() == 0){
    std::cout << "Estimation vector size is zero" << std::endl;
    return rmse;
  }
    
  //  * the estimation vector size should equal ground truth vector size
  if(estimations.size() == ground_truth.size()){
	VectorXd temp(4);

	//accumulate squared residuals
	for(int i=0; i < estimations.size(); ++i){
    		temp = estimations[i]-ground_truth[i];
    		temp = temp.array()*temp.array();
    		rmse += temp;
	}

	//calculate the mean
	rmse = rmse/estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();
  }
  else
    	std::cout << "Estimation vector and ground truth vector are of different size" << std::endl;

  //return the result
  return rmse;
}


MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
      * Calculate a Jacobian.
  */
  MatrixXd Hj_(3,4);

  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float c1 = px*px + py*py; 
  // make sure to not divide by 0
  if(c1 < 0.001 ){
	c1 = 0.001;
	//std::cout << "px and py close to zero" << std::endl;
  } 
  float c2 = sqrt(c1);
  float c3 = c1*c2;

  Hj_ << px/c2, py/c2, 0, 0,
        -py/c1, px/c1, 0, 0,
         py*(vx*py-vy*px)/c3, px*(px*vy-py*vx)/c3, px/c2, py/c2; // Jacobian

  return Hj_;

}
