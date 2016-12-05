#ifndef __RANDOMTREECLASSIFIERHPP__
#define __RANDOMTREECLASSIFIERHPP__

#include "RandomTree.hpp"

namespace gsm
{
  template<class T_X, class T_y>
  class RandomTreeClassifier : public RandomTree<T_X, T_y>
  {
  public:
    void train(
	       const Eigen::Matrix<T_X, Eigen::Dynamic, Eigen::Dynamic> &X,
	       const Eigen::Matrix<T_y, Eigen::Dynamic, Eigen::Dynamic> &y
	       ) override;
  };
}

#endif

template<class T_X, class T_y>
void
gsm::RandomTreeClassifier<T_X, T_y>::train(
				    const Eigen::Matrix<T_X, Eigen::Dynamic, Eigen::Dynamic> &X,
				    const Eigen::Matrix<T_y, Eigen::Dynamic, Eigen::Dynamic> &y
				    )
{
  // Compute the entropy to divide the training data into left and right
  // samples.
}
					 
