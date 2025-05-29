import os
import json
import requests
from pinecone import Pinecone
from dotenv import load_dotenv
from typing import List, Dict, Set
import streamlit as st
from llm import LLM

# Load environment variables
load_dotenv()

def identify_pdf_names_from_party_or_case(query: str) -> List[str]:
    '''
    Give pdf, case_number_array and party_name_array to llm and 
    ask it to return list of pdf file names that match.
    '''
    try:
        # # Initialize Pinecone
        # pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        # index = pc.Index("court-judgments")
        
        # Step 1: Fetch unique combinations from Pinecone
        unique_data = fetch_unique_combinations('metadata.json')
        
        if not unique_data:
            return []
        
        # Step 3: Get LLM to identify specific case numbers and party names
        identified_items = identify_case_and_party_names(
            query, 
            unique_data['case_numbers'], 
            unique_data['party_names']
        )
        
        # Step 4: Search Pinecone for matching records and get PDF filenames
        pdf_filenames = search_and_get_pdf_filenames(
            identified_items.get('case_numbers',[]),
            identified_items.get('party_names',[]),
            "metadata.json"
        )
        
        return pdf_filenames
        
    except Exception as e:
        print(f"Error in identify_pdf_names_from_party_or_case: {str(e)}")
        return []

def fetch_unique_combinations(file_path: str = "metadata.json") -> Dict:
    '''
    Fetch unique combinations of case_numbers and party_names from metadata.json
    '''
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)

        all_case_numbers = set()
        all_party_names = set()

        for pdf_metadata in metadata_dict.values():
            # Extract case numbers
            case_numbers = pdf_metadata.get('case_numbers', [])
            if isinstance(case_numbers, list):
                all_case_numbers.update(case_numbers)

            # Extract party names
            party_names = pdf_metadata.get('party_names', [])
            if isinstance(party_names, list):
                all_party_names.update(party_names)

        return {
            'case_numbers': list(all_case_numbers),
            'party_names': list(all_party_names)
        }

    except Exception as e:
        print(f"Error fetching unique combinations: {str(e)}")
        return {'case_numbers': [], 'party_names': []}

def check_user_mentions(query: str) -> Dict[str, bool]:
    '''
    Use LLM to check if user mentioned case numbers or party names
    '''
    try:
        # Initialize LLM without specifying provider or model (random selection)
        llm = LLM()
        
        prompt = f"""
        Analyze the following user query and determine if it mentions:
        1. Case numbers (legal case identifiers, docket numbers, etc.)
        2. Party names (names of individuals, companies, organizations involved in legal cases)
        
        Query: "{query}"
        
        Respond with a JSON object containing:
        {{
            "has_case_number": true/false,
            "has_party_name": true/false,
            "explanation": "Brief explanation of your analysis"
        }}
        """
        
        # Set prompts and make the request
        llm.system_prompt = "You are a legal assistant that analyzes queries for case numbers and party names."
        llm.user_prompt = prompt
        
        # Get response from LLM
        response = llm.ask()
        
        # Extract JSON from response text
        try:
            content = response['text']
            # Find JSON in the response
            start = content.find('{')
            end = content.rfind('}') + 1
            json_str = content[start:end]
            result = json.loads(json_str)
            
            # Add metadata to result
            result['llm_provider'] = response['provider']
            result['llm_model'] = response['model']
            result['api_key_name'] = response['api_key_name']
            
            return result
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Error parsing JSON from LLM response: {str(e)}")
            # Fallback: simple keyword detection
            return _fallback_detection(query)
            
    except Exception as e:
        print(f"Error in check_user_mentions: {str(e)}")
        # Fallback logic
        return _fallback_detection(query)

def _fallback_detection(query: str) -> Dict[str, bool]:
    """Fallback keyword-based detection when LLM fails"""
    return {
        "has_case_number": any(keyword in query.lower() for keyword in ['case', 'docket', 'number', 'v.', 'vs']),
        "has_party_name": any(keyword in query.lower() for keyword in ['plaintiff', 'defendant', 'petitioner', 'respondent', 'company', 'corp', 'party', 'private', 'limited']),
        "explanation": "Fallback keyword detection used due to LLM error",
        "llm_provider": "fallback",
        "llm_model": "keyword_detection",
        "api_key_name": "none"
    }

def identify_case_and_party_names(query: str, all_case_numbers: List[str], all_party_names: List[str]) -> Dict:
    '''
    Use LLM to identify specific case numbers and party names from the query
    '''
    try:
        # Initialize LLM without specifying provider or model (random selection)
        llm = LLM()
        
        prompt = f"""
        Given the user query and the available case numbers and party names from our database, 
        identify which specific case numbers and party names the user is referring to.
        
        User Query: "{query}"
        
        Available Case Numbers: {all_case_numbers}
        Available Party Names: {all_party_names}
        
        Instructions:
        - Be flexible with formatting (e.g., "Case 123" might match "123-CV-2023")
        - Consider partial name matches for party names
        - Return empty arrays if no matches found
        
        Respond with a JSON object:
        {{
            "case_numbers": ["matched_case_1", "matched_case_2"],
            "party_names": ["matched_party_1", "matched_party_2"]
        }}
        """
        
        # Set prompts and make the request
        llm.system_prompt = "You are a legal assistant that matches user queries to specific case numbers and party names from a database."
        llm.user_prompt = prompt
        
        # Get response from LLM
        response = llm.ask()
        
        # Extract JSON from response text
        try:
            content = response['text']
            # Find JSON in the response
            start = content.find('{')
            end = content.rfind('}') + 1
            json_str = content[start:end]
            result = json.loads(json_str)
            
            # Add metadata to result
            result['llm_provider'] = response['provider']
            result['llm_model'] = response['model']
            result['api_key_name'] = response['api_key_name']
            
            # Ensure required keys exist with default values
            if 'case_numbers' not in result:
                result['case_numbers'] = []
            if 'party_names' not in result:
                result['party_names'] = []
                
            return result
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Error parsing JSON from LLM response: {str(e)}")
            return _fallback_empty_result()
            
    except Exception as e:
        print(f"Error in identify_case_and_party_names: {str(e)}")
        return _fallback_empty_result()

def _fallback_empty_result() -> Dict:
    """Returns empty result with metadata when LLM fails"""
    return {
        "case_numbers": [],
        "party_names": [],
        "llm_provider": "fallback",
        "llm_model": "none",
        "api_key_name": "none",
        "error": "LLM processing failed, returned empty results"
    }

def search_and_get_pdf_filenames(case_numbers: List[str], party_names: List[str], file_path: str = "metadata.json") -> List[str]:
    '''
    Search metadata.json for records matching case numbers or party names and return unique PDF filenames
    '''
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)

        pdf_filenames = set()

        for pdf_filename, metadata in metadata_dict.items():
            # Check if any of the identified case numbers match
            record_case_numbers = metadata.get('case_numbers', [])
            case_match = any(
                case_num in record_case_numbers
                for case_num in case_numbers
            ) if case_numbers else False

            # Check if any of the identified party names match (partial or full, case-insensitive)
            record_party_names = metadata.get('party_names', [])
            party_match = any(
                any(party_name.lower() in record_party.lower() or record_party.lower() in party_name.lower()
                    for record_party in record_party_names)
                for party_name in party_names
            ) if party_names else False

            # If there's a match, add the PDF filename
            if case_match or party_match:
                pdf_filenames.add(pdf_filename)

        print(list(pdf_filenames))
        print('\n\n\n\n\n\n\n\n')

        return list(pdf_filenames)

    except Exception as e:
        print(f"Error in search_and_get_pdf_filenames: {str(e)}")
        return []