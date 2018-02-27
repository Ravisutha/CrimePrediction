#!/usr/bin/python3

class Path:
    """ Dataset path for each year """

    def __init__(self):
        """ Accept the year for which dataset path is required. """

        pass
        
    def _dataset_path (self, year=2015):
        """ Compile dataset path. """

        path_year = "../../Data/" + str (year)

        self.path_crime = path_year + "/crimes_" + str (year) + ".csv"

        self.path_sanity = path_year + "/sanitation_community_" + str(year) + ".csv"

        self.path_school = path_year + "/school_" + str(year) + ".csv"

        self.path_vehicles = path_year + "/vehicles_" + str(year) + ".csv"

        self.path_pot_holes = path_year + "/pot_holes_" + str(year) + ".csv"

        self.path_lights_one = path_year + "/lights_one_" + str(year) + ".csv"

        self.path_lights_all = path_year + "/lights_all_" + str(year) + ".csv"

        self.path_lights_alley = path_year + "/lights_alley_" + str(year) + ".csv"

        self.path_trees = path_year + "/trees_" + str(year) + ".csv"

        self.path_vacant = path_year + "/map_vacant_" + str(year) + ".csv"

        self.path_library = "../../Data/Static/map_libraries.csv"

        self.path_police = "../../Data/Static/Map_police_community.csv"

        self.path_IUCR = "../../Data/Static/IUCR.csv"

        self.path_community = "../../Data/Static/community.csv"

        self.path_output = path_year + "/network_" + str (year) + ".graphml"

    def get_path (self, year=2015, type="crime"):
        """ Get path for given year and type of network. """

        path = []

        #Generate dataset paths
        self._dataset_path (year)

        #Return set of dataset paths based on type of network
        if (type == "crime"):
            path.append (self.path_crime)
            path.append (self.path_IUCR)
            path.append (self.path_community)

        elif (type == "police"):
            path.append (self.path_police)

        elif (type == "sanity"):
            path.append (self.path_sanity)

        elif (type == "vehicles"):
            path.append (self.path_vehicles)

        elif (type == "pot_holes"):
            path.append (self.path_pot_holes)

        elif (type == "lights_one"):
            path.append (self.path_lights_one)

        elif (type == "lights_all"):
            path.append (self.path_lights_all)

        elif (type == "lights_alley"):
            path.append (self.path_lights_alley)

        elif (type == "trees"):
            path.append (self.path_trees)

        elif (type == "library"):
            path.append (self.path_library)

        elif (type == "vacant"):
            path.append (self.path_vacant)

        elif (type == "output"):
            path.append (self.path_output)

        elif (type == "school"):
            path.append (self.path_school)

        else:
            raise ValueError ("Don't have the path for newtork type: {}".format (type))

        return (path)

if (__name__ == "__main__"):
    """ Only execute if this file is run independently. (Don't run when imported). """

    path = Path ()
    
    print (path.get_path (2015, "sanity"))
    print (path.get_path (2015, "library"))
    print (path.get_path (2015, "police"))
    print (path.get_path (2015, "crime"))
    print (path.get_path (2015, "output"))
    print (path.get_path (2015, "lights_one"))
    print (path.get_path (2015, "lights_all"))
    print (path.get_path (2015, "lights_alley"))
    print (path.get_path (2015, "vacant"))
