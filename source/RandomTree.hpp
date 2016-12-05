#ifndef __RANDOMTREEHPP__
#define __RANDOMTREEHPP__
#include <iostream>
#include <memory>
#include <vector>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

namespace gsm
{
  enum Task
    {
      CLASSIFICATION,
      LINEAR_REGRESSION
    };
  
  template<class T>
  struct node
  {
    T _threshold;
    std::unique_ptr<node> _left;
    std::unique_ptr<node> _right;
    node(const T &threshold);
  };

  template<class T_X, class T_y>
  class RandomTree
  {
  protected:
    std::unique_ptr<node<T_X>> _root;
    Task                       _task;
    void printRandomTreeNode(const node<T_X> *nodeData);
  public:
    RandomTree();
    RandomTree(Task task);
    void addNode(const T_X &threshold);
    void printRandomTreeNode();

    // Make Train a Pure Virtual Function to be implemented by its derived class
    // i.e. RandomTreeClassifier or RandomTreeRegressor
    virtual void train(
		       const Eigen::Matrix<T_X, Eigen::Dynamic, Eigen::Dynamic> &X,
		       const Eigen::Matrix<T_y, Eigen::Dynamic, Eigen::Dynamic> &y
		       ) = 0;
  };
}
#endif

template<class T>
gsm::node<T>::node(const T &threshold) : _threshold(threshold), _left(nullptr), _right(nullptr)
{
}

template<class T_X, class T_y>
gsm::RandomTree<T_X, T_y>::RandomTree() :
  _root(nullptr),
  _task(gsm::Task::CLASSIFICATION)
{
}

template<class T_X, class T_y>
gsm::RandomTree<T_X, T_y>::RandomTree(gsm::Task task) :
  _root(nullptr),
  _task(task)
{
}

template<class T_X, class T_y>
void gsm::RandomTree<T_X, T_y>::addNode(const T_X &threshold)
{
  if (!_root)
    {
      _root = std::make_unique<node<T_X>>(threshold);
    }
  else
    {
      gsm::node<T_X> *tempNode = _root.get();

      while(tempNode->_left && tempNode->_right)
	{
	  if (tempNode->_threshold > threshold)
	    {
	      tempNode = tempNode->_left.get();
	    }
	  else
	    {
	      tempNode = tempNode->_right.get();
	    }
	}

      if (tempNode->_threshold > threshold)
	{
	  tempNode->_left = std::make_unique<node<T_X>>(threshold);
	}
      else
	{
	  tempNode->_right = std::make_unique<node<T_X>>(threshold);
	}
    }
}

template<class T_X, class T_y>
void gsm::RandomTree<T_X, T_y>::printRandomTreeNode()
{
  gsm::node<T_X> *tempNode = _root.get();
  printRandomTreeNode(tempNode);
}

template<class T_X, class T_y>
void gsm::RandomTree<T_X, T_y>::printRandomTreeNode(const gsm::node<T_X> *nodeData)
{
  if (nodeData == nullptr)
    {
      return;
    }

  printRandomTreeNode(nodeData->_left.get());
  std::cerr << nodeData->_threshold << std::endl;
  printRandomTreeNode(nodeData->_right.get());
}

