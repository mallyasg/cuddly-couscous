#include "RandomTreeClassifier.hpp"
#include <fstream>
#include <string>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

template<typename T_X, typename T_y>
void loadData(const std::string &fileName,
	      Eigen::Matrix<T_X, Eigen::Dynamic, Eigen::Dynamic> &X,
	      Eigen::Matrix<T_y, Eigen::Dynamic, Eigen::Dynamic> &y
	      )
{
  std::vector<std::vector<T_X>> trainingFeatures;
  std::vector<std::vector<T_y>> trainingLabels;
  
  std::string line;
  std::ifstream fileContainingTrainingData(fileName);
  std::string delimiter = " \t ";
  
  if (fileContainingTrainingData.is_open())
    {
      while(std::getline(fileContainingTrainingData, line))
	{
     	  // Split the line with tab as the delimiter
	  std::vector<int> label;
	  size_t pos = line.find(delimiter);
	  
	  label.push_back(std::stoi(line.substr(0, pos)));
	  trainingLabels.push_back(label);
	  	  	  
	  line.erase(0, pos + delimiter.length());
	  	   
	  std::vector<double> trainingFeature;
       	  while((pos = line.find(delimiter)) != std::string::npos)
	    {
	      trainingFeature.push_back(std::stod(line.substr(0, pos)));
	      line.erase(0, pos + delimiter.length());
	    }
	  trainingFeature.push_back(std::stod(line));
	  trainingFeatures.push_back(trainingFeature);
	}
    }

  X.resize(trainingFeatures.size(), trainingFeatures.at(0).size());
  y.resize(trainingLabels.size(), trainingLabels.at(0).size());

  for (size_t i = 0; i < trainingFeatures.size(); i++)
    {
      for (size_t j = 0; j < trainingLabels.at(i).size(); j++)
	{
	  y(i, j) = trainingLabels.at(i).at(j);
	}

      for (size_t j = 0; j < trainingFeatures.at(i).size(); j++)
	{
	  X(i, j) = trainingFeatures.at(i).at(j);
	}
    }

  
}

int main(int argc, char **argv)
{
  // gsm::RandomTreeClassifier<int, int> tree;
  // tree.addNode(10);
  // tree.addNode(3);
  // tree.addNode(20);
  // tree.printRandomTreeNode();

  // Load data from exp1_n2.txt
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> X;
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> y;
  
  loadData(std::string(argv[1]), X, y);
  
  if (X.rows() != y.rows())
    {
      exit(EXIT_FAILURE);
    }

  double maxX = FLT_MIN;
  double maxY = FLT_MIN;
  
  for (int i = 0; i < X.rows(); i++)
    {
      if (maxX < X(i, 0))
	{
	  maxX = X(i, 0);
	}

      if (maxY < X(i, 1))
	{
	  maxY = X(i, 1);
	}
    }

  cv::Mat imageTrainingPoints(500, 500, CV_8UC3);

  for (int i = 0; i < X.rows(); i++)
    {
      cv::Point2d center;
      center.x = X(i, 0) / maxX * imageTrainingPoints.cols;
      center.y = X(i, 1) / maxX * imageTrainingPoints.rows;
      if (y(i, 0) == 1)
	{  
	  cv::circle(
		     imageTrainingPoints,
		     center,
		     2.0,
		     cv::Scalar(255, 255, 255)
		     );
	}
      else
	{
	  cv::rectangle(
			imageTrainingPoints,
			cv::Point2d(center.x - 1, center.y - 1),
			cv::Point2d(center.x + 1, center.y + 1),
			cv::Scalar(255, 255, 255)
			);
	}
    }
  
    cv::imshow("Image", imageTrainingPoints);
    cv::waitKey(0);
  
  return 0;
}
