from conan import ConanFile
from conan.tools.cmake import cmake_layout


class Recipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"

    def requirements(self):
        self.requires("fmt/11.1.4")
        self.requires("gtest/1.16.0")

    def layout(self):
        cmake_layout(self)
