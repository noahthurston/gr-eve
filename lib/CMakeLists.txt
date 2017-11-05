# Copyright 2011,2012,2016 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# GNU Radio is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# GNU Radio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Radio; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.

########################################################################
# Setup library
########################################################################
include(GrPlatform) #define LIB_SUFFIX

include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

list(APPEND eve_sources
)

set(eve_sources "${eve_sources}" PARENT_SCOPE)
if(NOT eve_sources)
	MESSAGE(STATUS "No C++ sources... skipping lib/")
	return()
endif(NOT eve_sources)

add_library(gnuradio-eve SHARED ${eve_sources})
target_link_libraries(gnuradio-eve ${Boost_LIBRARIES} ${GNURADIO_ALL_LIBRARIES})
set_target_properties(gnuradio-eve PROPERTIES DEFINE_SYMBOL "gnuradio_eve_EXPORTS")

if(APPLE)
    set_target_properties(gnuradio-eve PROPERTIES
        INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib"
    )
endif(APPLE)

########################################################################
# Install built library files
########################################################################
include(GrMiscUtils)
GR_LIBRARY_FOO(gnuradio-eve RUNTIME_COMPONENT "eve_runtime" DEVEL_COMPONENT "eve_devel")

########################################################################
# Build and register unit test
########################################################################
include(GrTest)

include_directories(${CPPUNIT_INCLUDE_DIRS})

list(APPEND test_eve_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/test_eve.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/qa_eve.cc
)

add_executable(test-eve ${test_eve_sources})

target_link_libraries(
  test-eve
  ${GNURADIO_RUNTIME_LIBRARIES}
  ${Boost_LIBRARIES}
  ${CPPUNIT_LIBRARIES}
  gnuradio-eve
)

GR_ADD_TEST(test_eve test-eve)

########################################################################
# Print summary
########################################################################
message(STATUS "Using install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Building for version: ${VERSION} / ${LIBVER}")

