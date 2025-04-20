**California Road Network Analysis**
=====================================

**Project Overview**
--------------------

This project analyzes the California road network using network science techniques to identify critical congestion points, optimize routing strategies, and provide insights into urban planning decisions. The study aims to enhance overall connectivity and efficiency of the network.

**Project Abstract**
--------------------

This project investigates the California road network, a large-scale transportation network, to analyze urban traffic dynamics and infrastructure efficiency. The study identifies critical congestion points, optimizes routing strategies, and provides insights into how urban planning decisions can enhance overall connectivity.

**Network Details**
-------------------

* **Type of Network**: Transportation network
* **Nodes**: Intersections where two or more roads converge
* **Edges**: Road segments connecting intersections
* **Number of Nodes**: Approximately 1,965,206 intersections
* **Number of Edges**: Approximately 2,766,607 road segments

**Source of Network Data**
---------------------------

The data for this project is sourced from the Stanford Large Network Dataset Collection.

**Motivation and Objectives**
-----------------------------

* **Motivation**: The California road network presents a compelling case for analysis due to its scale and complexity.
* **Objectives**:
	1. Traffic Congestion Analysis: Identify key intersections and road segments that act as congestion hotspots.
	2. Routing Optimization: Develop and test algorithms to optimize routing, thereby reducing travel times and alleviating congestion.
	3. Infrastructure Efficiency: Evaluate how the current road network design impacts overall efficiency and identify areas for potential improvement.
	4. Urban Planning Insights: Provide recommendations for urban planners based on the network's structural characteristics and traffic patterns.

**Project Structure**
---------------------

This project consists of several modules:

* `app.py`: The main application file that sets up the Streamlit app.
* `pages`: A directory containing separate pages for different analyses, including:
	+ `02_congestion_analysis.py`: Congestion analysis page.
	+ `03_routing_optimization.py`: Routing optimization page.
	+ `04_urban_planning.py`: Urban planning page.

**Usage**
---------

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the application using `streamlit run app.py`.
4. Navigate to the different pages to perform various analyses.

**Contributors**
----------------

* Ishva Patel (21110082)
* Shubh Agarwal (21110205)

**License**
----------

This project is licensed under the MIT License. See `LICENSE` for details.