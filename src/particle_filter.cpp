/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    if (!is_initialized)
    {
        num_particles = 10;

        //Initilaize a random engine names "gen"
        default_random_engine gen;

        // creates a normal (Gaussian) distribution for x,y and theta
        normal_distribution<double> dist_x(x, std[0]);
        normal_distribution<double> dist_y(y, std[1]);
        normal_distribution<double> dist_psi(theta, std[2]);

        for (int i = 0; i < num_particles; ++i)
        {
            Particle particleTemp;
            particleTemp.id = i;
            particleTemp.x = dist_x(gen);
            particleTemp.y = dist_y(gen);
            particleTemp.theta = dist_psi(gen);
            particleTemp.weight = 1.0;
            particles.push_back(particleTemp);

            weights.push_back(1);
        }
        is_initialized = true;
    }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    //Initialize all the particles to the (first)GPS position as mean

    default_random_engine gen;

    for (int i = 0; i < num_particles; ++i)
    {
        if (fabs(yaw_rate) < 0.0001)
        {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }
        else
        {
            particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
            particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
            particles[i].theta += yaw_rate * delta_t;
        }

        std::normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
        std::normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
        std::normal_distribution<double> dist_psi(particles[i].theta, std_pos[2]);

        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_psi(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    for (int i = 0; i < observations.size(); i++)
    {
        double min_length = numeric_limits<double>::max();
        int index = -1;

        //// applying nearest neighbour technique on all landmarks in range
        for (int j = 0; j < predicted.size(); j++)
        {
            double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
            if (distance < min_length)
            {
                min_length = distance;
                index = j;
                //// observation id will be nearest in range landmark's id
                observations[i].id = predicted[index].id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks)
{
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    double s_x = std_landmark[0];     //std deviation of the meas in x
    double s_y = std_landmark[0];     //std deviation of the meas in y
    double s_theta = std_landmark[1]; //std deviation of the meas in theta

    for (int i = 0; i < num_particles; ++i)
    {
        std::vector<LandmarkObs> tObservations;     //Transformed Observations Vector
        std::vector<LandmarkObs> landmarks_inrange; //Hold Landmarks in sensor Range

        // Filter Valid Predictions based on sensor Range and distance between particle and Landmark
        for (int k = 0; k < map_landmarks.landmark_list.size(); ++k)
        {
            if (dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f) < sensor_range)
            {
                landmarks_inrange.push_back(LandmarkObs{
                    map_landmarks.landmark_list[k].id_i, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f,
                });
            }
        }

        for (int j = 0; j < observations.size(); j++)
        {
            LandmarkObs t_obs; //  Transformed Observation for Single particle

            t_obs.x = cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y + particles[i].x;
            t_obs.y = sin(particles[i].theta) * observations[j].x + cos(particles[i].theta) * observations[j].y + particles[i].y;
            t_obs.id = observations[j].id;
            tObservations.push_back(t_obs);
        }

        //Find the prediction that points to a Observation, finds the probability/Weights
        dataAssociation(landmarks_inrange, tObservations);

        double total_prob = 1.0;
        double temp_prob = 1.0;

        //std::cout<<"---------------------Weights Calc------------------------"<<std::endl;

        for (int m = 0; m < tObservations.size(); m++)
        {
            double lm_x;
            double lm_y;
            int lm_id;
            double p_x = tObservations[m].x;
            double p_y = tObservations[m].y;

            for (int idx = 0; idx < landmarks_inrange.size(); idx++)
            {
                if (tObservations[m].id == landmarks_inrange[idx].id)
                {
                    //Find the nearest landmark for the observation and use that landmark instead of the observation
                    lm_x = landmarks_inrange[idx].x;
                    lm_y = landmarks_inrange[idx].y;
                    ////landmark's id and observation's id will be same here
                    lm_id = landmarks_inrange[idx].id;
                }
            }
            // Calculate Multivariate-Gaussian Probability for each observations(measurements)
            double d_x = lm_x - p_x;
            double d_y = lm_y - p_y;
            //Total probability is the product of individual measurement Probabilities.
            temp_prob = (1.0 / (2.0 * M_PI * s_x * s_y)) * exp(-(((d_x * d_x) / (2 * s_x * s_x)) + ((d_y * d_y) / (2 * s_y * s_y))));
            total_prob *= temp_prob;
        }
        particles[i].weight = total_prob;
        weights[i] = total_prob;
    }
}

void ParticleFilter::resample()
{
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::discrete_distribution<int> d(weights.begin(), weights.end());       // Define a discrete distribution
    std::vector<Particle> resampled_particles;                               // Resampled particles holder
    std::default_random_engine gen;

    for (int i = 0; i < num_particles; i++)
    {
        auto index = d(gen);
        resampled_particles.push_back(std::move(particles[index]));
    }
    //assign the particles from holder to the original
    particles = std::move(resampled_particles);
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}