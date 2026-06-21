import networkx as nx
import numpy as np

class ResourceOptimizer:
    def __init__(self):
        pass

    def allocate_resources(self, impact_forecast):
        """
        Calculates optimal manpower and barricades based on forecasted severity.
        """
        duration = impact_forecast['predicted_duration_mins']
        radius = impact_forecast['congestion_radius_meters']
        
        # Base requirements
        manpower = 2
        
        # Scale with radius (e.g., 1 cop per 400m radius)
        manpower += int(radius / 400)
        
        # Scale with duration (relief shifts)
        if duration > 180:
            manpower = int(manpower * 1.5)
        elif duration > 360:
            manpower = int(manpower * 2.0)
            
        barricades = manpower * 3
        
        return {
            'recommended_personnel': max(2, manpower),
            'recommended_barricades': max(4, barricades)
        }
        
    def calculate_diversion(self, event_lat, event_lon, radius_meters):
        """
        Simulates dynamic routing. Recommends alternative waypoints around the congestion.
        """
        # 1 degree lat is approx 111 km. 
        offset_deg = (radius_meters / 111000.0) * 1.3 # 30% margin around radius
        
        diversion_routes = [
            {"lat": event_lat + offset_deg, "lon": event_lon + offset_deg, "instruction": "Divert North-East via alternative arterial"},
            {"lat": event_lat + offset_deg, "lon": event_lon - offset_deg, "instruction": "Divert North-West via alternative arterial"},
            {"lat": event_lat - offset_deg, "lon": event_lon + offset_deg, "instruction": "Divert South-East via alternative arterial"},
            {"lat": event_lat - offset_deg, "lon": event_lon - offset_deg, "instruction": "Divert South-West via alternative arterial"},
        ]
        
        return diversion_routes
