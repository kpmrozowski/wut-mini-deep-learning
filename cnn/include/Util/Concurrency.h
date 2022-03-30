#ifndef UTIL_CONCURRENCY_H
#define UTIL_CONCURRENCY_H
#include <fmt/core.h>
#pragma once
#ifdef WITH_CUDA
#include <Eden_resources/Ngpus_Ncpus.h>
#endif
#include "augumentation.h"
#include <condition_variable>

namespace regularization {

enum class regularization_type {
    none,
    l1,
    l2,
};

}

typedef std::tuple<
    regularization::regularization_type,
    double,
    augumentation::augumentation_type,
    std::string,
    int> SimulationSetting;


class join_threads {
   std::vector<std::thread> &threads;
   std::atomic<int>& m_workers_up;
   bool joined_earlier = false;
   
   void join_all() {
      fmt::print("join_all: m_workers_up={}", m_workers_up);
      for (unsigned long i = 0; i < threads.size(); ++i) {
         while (true) {
            if (threads[i].joinable()){
               threads[i].join();
               break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            fmt::print("join_all: m_workers_up={}", m_workers_up);
         }
      }
      fmt::print("client_threads all threads joined!");
   }
 public:
   explicit join_threads(std::vector<std::thread> &threads_, std::atomic<int>& workers_up)
    : threads(threads_)
    , m_workers_up(workers_up) {}
   ~join_threads() {
      if (joined_earlier) return;
      join_all();
   }
   void join_earlier() {
      joined_earlier = true;
      join_all();
   }
};

class client_threads {
   std::atomic_bool m_done;
   std::vector<std::thread> m_threads;
   std::atomic<int> m_workers_up = 0;
   join_threads m_joiner;
   void client_work(int run_idx);

 public:
   std::string imagenette_data_path;
   SimulationSetting setting;
   client_threads(unsigned cpus_count, SimulationSetting s, std::string data_path = "../../../cifar-10")
       : m_done(false)
       , m_joiner(m_threads, m_workers_up)
       , imagenette_data_path(data_path)
       , setting(s)
   {
#ifdef WITH_CUDA
      if (cpus_count == 0)
         cpus_count = Eden_resources::get_cpus_count();
#endif
      fmt::print("creating {} client threads", cpus_count);
      try {
         for (unsigned i = 0; i < cpus_count; ++i) {
            m_threads.push_back(std::thread(&client_threads::client_work, this, i));
            fmt::print("83");
            // m_threads[i].detach();
         }
      } catch (...) {
         m_done = true;
         throw;
      }
   }
   ~client_threads() {
      m_done = true;
      fmt::print("client_threads destroyed!");
   }
   void join_clients() {
      m_joiner.join_earlier();
   }
};

#endif // CARCASSONNE_RL_CONCURRENCY_H
