#!/usr/bin/env python3
"""
Simple Neo4j connectivity test
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

def test_simple_connection():
    """Test basic Neo4j connectivity"""
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME") 
    password = os.getenv("NEO4J_PASSWORD")
    
    print(f"Testing connection to: {uri}")
    print(f"Username: {username}")
    print(f"Password: {'*' * len(password) if password else 'None'}")
    
    try:
        with GraphDatabase.driver(uri, auth=(username, password)) as driver:
            driver.verify_connectivity()
            print("✅ Connection successful!")
            
            # Test simple query
            records, summary, keys = driver.execute_query(
                "RETURN 'Hello Neo4j!' as message",
                database_="neo4j"
            )
            
            for record in records:
                print(f"Response: {record.data()}")
                
            return True
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_simple_connection()