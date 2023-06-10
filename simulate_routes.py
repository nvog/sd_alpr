# Copyright (c) 2023 Nikolai Vogler

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import geopy
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import argparse
import random
import asyncio
import os
import aiolimiter
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# SERVER="https://router.project-osrm.org"
SERVER="http://savernake.ucsd.edu:5000"


def get_route(start_location, end_location):
    """ Get the route using Open Source Routing API
    :param start_location: string of start location lon, lat
    :param end_location: string of end location lon, lat
    :return: list of tuples of (lat, lon) coordinates along route
    """
    geolocator = Nominatim(user_agent="alpr")  # initialize geolocator
    # look up coordinates
    start_location = geolocator.reverse(start_location)
    end_location = geolocator.reverse(end_location)
    start_coords = (start_location.latitude, start_location.longitude)
    end_coords = (end_location.latitude, end_location.longitude)

    # get the route using Open Source Routing API
    url = f"{SERVER}/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}?overview=full&alternatives=false&annotations=nodes&geometries=geojson"
    print(url)
    response = requests.get(url)
    route_data = response.json()
    lat_lons = [(lat, lon) for lon, lat in route_data['routes'][0]['geometry']['coordinates']]
    return lat_lons


def haversine_distance(lat1, lon1, lat2, lon2):                                                                                  
    ''' Returns the distance between the points in m. Vectorized and will work with arrays and return an array of       
    distances, but will also work with scalars and return a scalar.
    From: https://www.tjansson.dk/2021/03/vectorized-gps-distance-speed-calculation-for-pandas/
    :param lat1: latitude(s) of first point
    :param lon1: longitude(s) of first point
    :param lat2: latitude(s) of second point
    :param lon2: longitude(s) of second point
    '''                          
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])                                                  
    a = np.sin((lat2 - lat1) / 2.0)**2 + (np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0)**2)                 
    distance = 6371 * 2 * np.arcsin(np.sqrt(a))                                                                         
    return distance * 1000


def find_cameras_on_route_slow(route, alpr_coords, radius):
    """ Find cameras within radius of route (not vectorized with geodesic distance)
    :param route: list of tuples of (lat, lon) coordinates of route
    :param alpr_coords: list of tuples of (lat, lon) coordinates of ALPR cameras
    :param radius: radius in meters to check for cameras along route
    """
    valid_locations = set()
    for point in alpr_coords:
        for coord in route:
            distance = geodesic(coord, point).meters
            if distance <= radius:
                valid_locations.add(point)
    return valid_locations


def find_cameras_on_route(route, alpr_coords, radius):
    """ Find cameras within radius of route vectorized with Haversine over camera coordinates
    :param route: list of tuples of (lat, lon) coordinates of route
    :param alpr_coords: list of tuples of (lat, lon) coordinates of ALPR cameras
    :param radius: radius in meters to check for cameras along route
    """
    valid_locations = set()
    # loop thru each point on route and check if it's within radius of any alpr camera
    # TODO: this could be further vectorized, but seems to be fast enough for now
    for i in range(len(route)):
        route_lat, route_lon = route[i]
        alpr_lat, alpr_lon = np.array(alpr_coords)[:,0], np.array(alpr_coords)[:,1]
        distances = haversine_distance(route_lat, route_lon, alpr_lat, alpr_lon)
        valid_camera_indices = (distances <= radius).nonzero()[0]
        for j in valid_camera_indices:
            valid_locations.add(alpr_coords[j])
    return valid_locations


def plot_map(df_alpr, df_start, df_end, route):
    """ Use plotly to plot the route on a map along with camera locations.
    :param df_alpr: dataframe of ALPR camera locations
    :param df_start: dataframe of start locations
    :param df_end: dataframe of end locations
    :param route: list of tuples of (lat, lon) coordinates of route
    """
    # plot alpr locations
    fig = px.scatter_mapbox(
        df_alpr, 
        lat="Latitude", 
        lon="Longitude", 
        hover_data=["Latitude", "Longitude"],
        zoom=10, 
        height=900,
        width=1400
    )
    for i in range(len(route)-1):
        fig.add_trace(
                go.Scattermapbox(
                    mode='lines',
                    lon=[route[i][1], route[i+1][1]],
                    lat=[route[i][0], route[i+1][0]],
                    line_color='green',
                )
            )

    fig.update(layout_showlegend=True)
    fig.update_layout(mapbox_style="open-street-map")
    # fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    # fig.update_traces(line=dict(color="green", width=20))
    fig.show()


def filter_destinations(df_end, destination_radius_threshold, destination_labels_whitelist):
    """ Filter end locations by radius and/or whitelist
    :param df_end: dataframe of end locations
    :param destination_radius_threshold: radius in meters to find destination or None to use all destinations
    :param destination_labels_whitelist: list of destination labels to filter on
    """
    before_len = len(df_end)
    if destination_labels_whitelist:
        df_end = df_end[df_end['place_label'].isin(destination_labels_whitelist)]
    if destination_radius_threshold:
        df_end = df_end[df_end['distance'] <= destination_radius_threshold]
    print(f"Filtered down {before_len} original end locations to {len(df_end)} after radius/whitelist filtering.")
    return df_end


def simulate_routes_serial(alpr_coords, df_start, df_end, n=1, camera_radius=25, destination_radius_threshold=4828, destination_labels_whitelist=None, start_district=None):
    """ Simulate n random routes from random start locations to random 
    end locations (within destination_radius of start location), 
    observing the number of cameras passed (within camera_radius) along the route.

    :param alpr_coords: list of tuples of (lat, lon) coordinates of ALPR cameras
    :param df_start: dataframe of start locations
    :param df_end: dataframe of end locations
    :param n: number of random routes to simulate
    :param camera_radius: radius in meters to check for cameras along route
    :param destination_radius: radius in meters to find destination or None to use all destinations
    :return: list of camera coordinates reached along route
    """    
    cameras_reached = []
    if start_district is not None:
        df_start = df_start[df_start['district'] == start_district]
    start_locations = df_start.sample(n=n, replace=True)
    for i in range(n):
        # reformat start coords as string
        start_location = str(start_locations.iloc[i].coordinate1) + ',' + str(start_locations.iloc[i].coordinate2)
        df_end_i = df_end.copy()
        # sort df_end_i's rows by the distance from the start_location
        df_end_i['distance'] = df_end_i.apply(lambda row: geodesic((row.coordinate1, row.coordinate2), (start_locations.iloc[i].coordinate1, start_locations.iloc[i].coordinate2)).meters, axis=1)
        df_end_i = df_end_i.sort_values(by=['distance'])
        df_end_i = filter_destinations(df_end_i, destination_radius_threshold, destination_labels_whitelist)
        if len(df_end_i) == 0:
            print(f'No end locations within {destination_radius_threshold}m of start location. Skipping...')
            continue
        print(f'Sampling from {len(df_end_i)} end locations within', destination_radius_threshold, 'meters of start location.')
        end_location = df_end_i.sample(n=1).iloc[0]
        # reformat end coords as str
        end_lat_lon = str(end_location.coordinate1) + ', ' + str(end_location.coordinate2)
        print(f'Computing random route #{i+1} from neighborhood at ({start_location}) to {end_location.place_name} ({end_lat_lon})')
        route = get_route(start_location, end_lat_lon)
        print('Route length (pts):', len(route))
        valid_locations = find_cameras_on_route(route, alpr_coords, camera_radius)
        print(f'Passed within {camera_radius}m of {len(valid_locations)} cameras along random route #{i+1}.')
        cameras_reached.append(len(valid_locations))
    return cameras_reached


async def _throttled_api_request(alpr_coords, start_location, df_end_i, camera_radius, destination_radius_threshold, destination_labels_whitelist, limiter, i):
    # reformat start coords as string
    start_lat_lon = str(start_location.coordinate1) + ',' + str(start_location.coordinate2)
    # sort df_end's rows by the distance from the start_location
    df_end_i['distance'] = df_end_i.apply(lambda row: geodesic((row.coordinate1, row.coordinate2), (start_location.coordinate1, start_location.coordinate2)).meters, axis=1)
    df_end_i = df_end_i.sort_values(by=['distance'])
    df_end_i = filter_destinations(df_end_i, destination_radius_threshold, destination_labels_whitelist)
    if len(df_end_i) == 0:
        print(f'No end locations within {destination_radius_threshold}m of start location. Skipping...')
        return None
    print(f'Sampling from {len(df_end_i)} end locations within', destination_radius_threshold, 'meters of start location.')
    end_location = df_end_i.sample(n=1).iloc[0]
    # reformat end coords as str
    end_lat_lon = str(end_location.coordinate1) + ', ' + str(end_location.coordinate2)
    print(f'Computing random route #{i+1} from neighborhood at ({start_lat_lon}) to {end_location.place_name} ({end_lat_lon})')
    route = get_route(start_lat_lon, end_lat_lon)
    print('Route length (pts):', len(route))
    valid_locations = find_cameras_on_route(route, alpr_coords, camera_radius)
    print(f'Passed within {camera_radius}m of {len(valid_locations)} cameras along random route #{i+1}.')
    return len(valid_locations)


async def simulate_routes(alpr_coords, df_start, df_end, n=1, camera_radius=18, destination_radius_threshold=4828, destination_labels_whitelist=None, start_district=None, requests_per_minute=5000):
    """ Simulate n random routes from random start locations to random 
    end locations (within destination_radius of start location), 
    observing the number of cameras passed (within camera_radius) along the route.

    :param alpr_coords: list of tuples of (lat, lon) coordinates of ALPR cameras
    :param df_start: dataframe of start locations
    :param df_end: dataframe of end locations
    :param n: number of random routes to simulate
    :param camera_radius: radius in meters to check for cameras along route
    :param destination_radius: radius in meters to find destination or None to use all destinations
    :return: list of camera coordinates reached along route
    """    
    if start_district is not None:
        df_start = df_start[df_start['district'] == start_district]
    start_locations = df_start.sample(n=n, replace=True)
    session = ClientSession()
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_api_request(alpr_coords, start_locations.iloc[i], df_end.copy(), camera_radius, destination_radius_threshold, destination_labels_whitelist, limiter, i)
        for i in range(n)
    ]
    # cameras_reached = await tqdm_asyncio.gather(async_responses, return_exceptions=True)
    errs = 0
    cameras_reached = []
    for done in tqdm_asyncio.as_completed(async_responses):
        try:
            cameras_reached.append(await done)
        except AsyncException:
            errs += 1
    print('Errors:', errs)
    await session.close()
    return [i for i in cameras_reached if i is not None]


async def _throttled_api_request_case_study(alpr_coords, start_location, case_study_destination, camera_radius, limiter, i):
    # reformat start coords as string
    start_lat_lon = str(start_location.coordinate1) + ',' + str(start_location.coordinate2)
    # sort alpr_coords by the distance from the case study destination
    alpr_coords = list(sorted(alpr_coords, key=lambda coord: geodesic(coord, case_study_destination).meters))
    print(f'Computing random route #{i+1} from neighborhood at ({start_lat_lon}) to case study destination {case_study_destination} ({case_study_destination})')
    route = get_route(start_lat_lon, case_study_destination)
    print('Route length (pts):', len(route))
    valid_locations = find_cameras_on_route(route, alpr_coords, camera_radius)
    print(f'Passed within {camera_radius}m of {len(valid_locations)} cameras along random route #{i+1}.')
    # find cameras within 1km of case study destination
    cameras_near_destination = [coord for coord in alpr_coords if geodesic(coord, case_study_destination).meters <= 1000]
    cameras_passed_near_destination = list(set(valid_locations).intersection(set(cameras_near_destination)))
    return cameras_passed_near_destination


async def simulate_routes_case_study(alpr_coords, df_start, case_study_destination, n=1, camera_radius=18, start_district=None, requests_per_minute=5000):
    """ Simulate n random routes from random start locations to particular case study 
    end locations, observing the number of cameras nearby the end location that were passed.
    Basically, this would show that it's possible for surveillance of travel to a particular destination 
    from many different start locations.

    :param alpr_coords: list of tuples of (lat, lon) coordinates of ALPR cameras
    :param df_start: dataframe of start locations
    :param df_end: dataframe of end locations
    :param n: number of random routes to simulate
    :param camera_radius: radius in meters to check for cameras along route
    :param destination_camera_radius: radius in meters to find destination or None to use all destinations
    :return: list of camera coordinates reached along route
    """    
    if start_district is not None:
        df_start = df_start[df_start['district'] == start_district]
    start_locations = df_start.sample(n=n, replace=True)
    session = ClientSession()
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_api_request_case_study(alpr_coords.copy(), start_locations.iloc[i], case_study_destination, camera_radius, limiter, i)
        for i in range(n)
    ]
    # cameras_reached = await tqdm_asyncio.gather(async_responses, return_exceptions=True)
    errs = 0
    cameras_reached = []
    for done in tqdm_asyncio.as_completed(async_responses):
        try:
            cameras_reached.append(await done)
        except AsyncException:
            errs += 1
    print('Errors:', errs)
    await session.close()
    return [i for i in cameras_reached if i is not None]


def make_argparser():
    parser = argparse.ArgumentParser(description='Simulate routes')
    parser.add_argument('--camera_csv', type=str, default='/Users/nvog/Downloads/ALPR_camera_map.csv', help='ALPR camera csv file.')
    parser.add_argument('--source_csv', type=str, default='/Users/nvog/Downloads/Neighborhood_Locations_All.csv', help='Source location csv file.')
    parser.add_argument('--destination_csv', type=str, default='/Users/nvog/Downloads/Sensitive_Destinations_All.csv', help='Destination location csv file.')
    parser.add_argument('--demographics_csv', type=str, default='/Users/nvog/Downloads/SANDAG_District_Demographic_Info.csv', help='District demographic csv file.')
    parser.add_argument('--camera_radius', type=int, default=18, help='Camera view radius (in meters). Chosen to be 18 meters empirically.')
    parser.add_argument('--destination_radius_threshold', type=int, default=4828, help='Upper bound for distance from destination to source (4828 is 3 mi in meters).')
    parser.add_argument('--start_district', type=int, default=None, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], help='Optional district number to simulate routes from.')
    parser.add_argument('--num_routes', type=int, default=1, help='Number of routes to simulate.')
    parser.add_argument('--num_routes_by_population_pct_total', type=int, default=100, help='Number of total routes to simulate across all districts (sampled as % of population).')
    parser.add_argument('--demographic', type=str, choices=["hispanic", "white", "black", "american_indian", "asian_pacific_isl", "all_other", "total"], default="total", help='Population demographic to choose from.')
    parser.add_argument('--seed', type=int, default=None, help='Optional seed for random number generator.')
    parser.add_argument('--destination_labels_whitelist', type=str, default=None, 
        choices=[
            "dispensary", "grocery_store", "hospital", 
            "gun store", "womens_health_clinic", "religious_site",
            "HIV_clinic", "abortion_clinic", "pharmacy", "airport", "school",
            "sensitive_only", "non-sensitive_only"
        ], 
        nargs='+',
        help='Optional place label(s) to visit destinations only with that place_label.'
    )
    parser.add_argument('--case_study_destination', type=str, default=None, help='Optional single case study destination in tuple (lat, lon) format.')
    return parser


if __name__ == '__main__':
    parser = make_argparser()
    args = parser.parse_args()
    print(args)
    print()

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    df_alpr = pd.read_csv(args.camera_csv)
    df_start = pd.read_csv(args.source_csv)
    df_end = pd.read_csv(args.destination_csv)
    df_dem = pd.read_csv(args.demographics_csv)

    # filter demographics by start district
    if args.start_district is not None:
        dem_pop_str = args.demographic + '_population'
        # calculate number of routes as the percentage of population for the demographic in the district
        num_routes = int(args.num_routes_by_population_pct_total * (df_dem[args.demographic + '_population'][df_dem['district'] == args.start_district].sum() / df_dem[args.demographic + '_population'].sum()))
        df_dem = df_dem[df_dem['district'] == args.start_district]

    alpr_lat_lons = df_alpr[['Latitude', 'Longitude']]
    alpr_coords = list(
        zip(
            alpr_lat_lons['Latitude'].tolist(), 
            alpr_lat_lons['Longitude'].tolist()
        )
    )
    if args.destination_labels_whitelist == ['sensitive_only']:
        destination_labels_whitelist = ["dispensary", "hospital", "gun_store", "womens_health_clinic", "religious_site", "HIV_clinic", "abortion_clinic"]
    elif args.destination_labels_whitelist == ['non-sensitive_only']:
        destination_labels_whitelist = ["grocery_store", "pharmacy", "airport", "school"]
    else:
        destination_labels_whitelist = args.destination_labels_whitelist

    print('> Destination labels whitelist:', destination_labels_whitelist)

    if args.case_study_destination is not None:
        print('> Case study destination:', args.case_study_destination)
        case_study_destination = [float(c) for c in args.case_study_destination.split(',')]
        camera_coords_reached = asyncio.run(simulate_routes_case_study(
            alpr_coords, 
            df_start, 
            args.case_study_destination, 
            n=args.num_routes,
            camera_radius=args.camera_radius,
            start_district=args.start_district
        ))
        cameras_reached = [c for cl in camera_coords_reached for c in cl]
        print(cameras_reached)
        plt.figure(figsize=(8, 10))
        # make a bar plot using cameras_reached as the data with the labels being the unique entries in labels and the values as percentages
        plt.bar(range(0, len(set(cameras_reached))), [cameras_reached.count(c) / len(cameras_reached) for c in set(cameras_reached)], color='blue')
        # plt.hist(cameras_reached, bins=range(0, max(cameras_reached)),  weights=np.ones(len(cameras_reached)) / len(cameras_reached))
        plt.xlabel('Cameras within 1km of destination')
        plt.xticks(range(0, len(set(cameras_reached))), [f'Camera\n({c[0]},\n{c[1]})' for c in set(cameras_reached)])
        plt.title(f'Hit rate on cameras around 1km radius of Little Italy\ndestination from simulated, random start locations (n={args.num_routes})')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.savefig(f'casestudy_nr{args.num_routes}_cr{args.camera_radius}.png')
    else:
        cameras_reached = asyncio.run(simulate_routes(
            alpr_coords, 
            df_start, 
            df_end, 
            n=args.num_routes,
            camera_radius=args.camera_radius,
            destination_radius_threshold=args.destination_radius_threshold,
            destination_labels_whitelist=destination_labels_whitelist,
            start_district=args.start_district
        ))
        print(cameras_reached)

        print(
            f'Average number of cameras passed per route ({len(cameras_reached)} total route(s)): ' 
            f'{np.mean(cameras_reached) if sum(cameras_reached) > 0 else "NaN"} +/- '
            f'{np.std(cameras_reached) if sum(cameras_reached) > 0 else "NaN"} ' 
            f'(median: {np.median(cameras_reached) if sum(cameras_reached) > 0 else "NaN"})'
        )

        plt.figure(figsize=(8, 10))
        # make a bar plot using cameras_reached as the data with the labels being the unique entries in labels and the values as percentages
        plt.bar(range(0, max(cameras_reached)), [cameras_reached.count(i) / len(cameras_reached) for i in range(0, max(cameras_reached))], color='blue')
        # plt.hist(cameras_reached, bins=range(0, max(cameras_reached)),  weights=np.ones(len(cameras_reached)) / len(cameras_reached))
        plt.xlabel('Number of cameras passed')
        plt.title(f'ALPR Cameras Passed on Simulated, Random Routes (n={args.num_routes})')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.savefig(f'hist_nr{args.num_routes}_cr{args.camera_radius}_sd{args.start_district}.png')
        
        # plot_map(df_alpr, df_start, df_end, route)