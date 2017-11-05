INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_EVE eve)

FIND_PATH(
    EVE_INCLUDE_DIRS
    NAMES eve/api.h
    HINTS $ENV{EVE_DIR}/include
        ${PC_EVE_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    EVE_LIBRARIES
    NAMES gnuradio-eve
    HINTS $ENV{EVE_DIR}/lib
        ${PC_EVE_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(EVE DEFAULT_MSG EVE_LIBRARIES EVE_INCLUDE_DIRS)
MARK_AS_ADVANCED(EVE_LIBRARIES EVE_INCLUDE_DIRS)

