#ifndef MSI_CARCASSONNE_CSVLOGGER_H
#define MSI_CARCASSONNE_CSVLOGGER_H
// #include <fmt/os.h>
#include <fmt/os.h>
#include <fmt/compile.h>
#include <string_view>
#include <sstream>

namespace util {

class CSVLogger {
   fmt::ostream m_file;
   std::string m_template;

 public:
   template<typename ...ARGS>
   explicit CSVLogger(std::string_view filename, ARGS ...args) noexcept : m_file(fmt::output_file(filename.data(), fmt::buffer_size=4096)) {
      std::stringstream stream;
      stream << "{}";
      for (int i = 1; i < sizeof...(ARGS); ++i) {
         stream << ",{}";
      }
      stream << '\n';
      m_template = stream.str();
      log(args...);
   }

   template<typename ...ARGS>
   void log(ARGS ...args) {
      m_file.print(m_template.c_str(), args...);
      m_file.flush();
   }
   void close() {
      m_file.close();
   }
};

}// namespace util

#endif//MSI_CARCASSONNE_CSVLOGGER_H
