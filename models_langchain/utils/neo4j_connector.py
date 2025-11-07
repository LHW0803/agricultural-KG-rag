from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import logging

class Neo4jConnector:
    """Neo4j database connector for Knowledge Graph operations"""
    
    def __init__(self, uri: str, username: str, password: str):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j database URI (e.g., "bolt://localhost:7687")
            username: Database username
            password: Database password
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        
    def connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            self.driver.verify_connectivity()
            print(f"Connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            return False
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            
    def query(self, cypher_query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        Execute a Cypher query
        
        Args:
            cypher_query: Cypher query string
            parameters: Query parameters
            
        Returns:
            Query results as list of dictionaries
        """
        with self.driver.session() as session:
            result = session.run(cypher_query, parameters or {})
            return [record.data() for record in result]
    
    def find_entity(self, entity_name: str, label: str = "HudongItem") -> Optional[Dict]:
        """Find an entity by name"""
        query = f"""
        MATCH (n:{label} {{title: $title}})
        RETURN n
        LIMIT 1
        """
        results = self.query(query, {"title": entity_name})
        return results[0] if results else None
    
    def get_entity_relations(self, entity_name: str, limit: int = 10) -> List[Dict]:
        """Get relations for a given entity"""
        query = """
        MATCH (n {title: $title})-[r:RELATION]->(m)
        RETURN n.title as source, type(r) as relation, r.type as relation_type, m.title as target
        LIMIT $limit
        """
        return self.query(query, {"title": entity_name, "limit": limit})
    
    def get_entity_attributes(self, entity_name: str) -> List[Dict]:
        """Get attributes for a given entity"""
        query = """
        MATCH (n {title: $title})-[r:RELATION]->(m)
        WHERE r.type IS NOT NULL
        RETURN n.title as entity, r.type as attribute_name, m.title as attribute_value
        """
        return self.query(query, {"title": entity_name})
    
    def search_similar_entities(self, keyword: str, limit: int = 5) -> List[Dict]:
        """Search for entities with similar names"""
        query = """
        MATCH (n)
        WHERE n.title CONTAINS $keyword
        RETURN n.title as title, labels(n) as labels
        LIMIT $limit
        """
        return self.query(query, {"keyword": keyword, "limit": limit})
    
    def get_shortest_path(self, entity1: str, entity2: str, max_length: int = 5) -> List[Dict]:
        """Find shortest path between two entities"""
        query = """
        MATCH path = shortestPath((n {title: $entity1})-[*..%d]-(m {title: $entity2}))
        RETURN [node in nodes(path) | node.title] as path,
               [rel in relationships(path) | type(rel)] as relations
        """ % max_length
        return self.query(query, {"entity1": entity1, "entity2": entity2})
    
    def get_entity_context(self, entity_name: str, depth: int = 2) -> Dict[str, Any]:
        """
        Get comprehensive context for an entity
        
        Args:
            entity_name: Name of the entity
            depth: How many hops to traverse
            
        Returns:
            Dictionary containing entity information and relations
        """
        # Get entity details
        entity = self.find_entity(entity_name)
        if not entity:
            return {}
        
        # Get direct relations
        relations = self.get_entity_relations(entity_name, limit=20)
        
        # Get attributes
        attributes = self.get_entity_attributes(entity_name)
        
        # Get connected entities for deeper context
        connected_query = """
        MATCH (n {title: $title})-[r*1..%d]-(m)
        RETURN DISTINCT m.title as connected_entity, labels(m) as labels
        LIMIT 30
        """ % depth
        connected = self.query(connected_query, {"title": entity_name})
        
        return {
            "entity": entity,
            "relations": relations,
            "attributes": attributes,
            "connected_entities": connected
        }
    
    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics"""
        node_count_query = "MATCH (n) RETURN count(n) as count"
        relation_count_query = "MATCH ()-[r]->() RETURN count(r) as count"
        
        nodes = self.query(node_count_query)[0]['count']
        relations = self.query(relation_count_query)[0]['count']
        
        # Count by label
        label_counts = {}
        for label in ['HudongItem', 'NewNode', 'Weather']:
            query = f"MATCH (n:{label}) RETURN count(n) as count"
            result = self.query(query)
            if result:
                label_counts[label] = result[0]['count']
        
        return {
            'total_nodes': nodes,
            'total_relations': relations,
            'label_counts': label_counts
        }