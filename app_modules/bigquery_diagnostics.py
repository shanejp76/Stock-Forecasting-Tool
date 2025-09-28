"""
Streamlit Cloud BigQuery Diagnostic Tool

This creates a simple diagnostic display that can be embedded in the main Streamlit app
to show detailed BigQuery connection status for troubleshooting production issues.
"""

import streamlit as st
import json
from datetime import datetime
from app_modules.bigquery_client import get_bigquery_client

def display_bigquery_diagnostics():
    """Display comprehensive BigQuery diagnostic information in Streamlit"""
    
    st.subheader("🔧 BigQuery Connection Diagnostics")
    
    with st.expander("Click to view detailed BigQuery diagnostics", expanded=False):
        
        # Initialize client and get diagnostics
        try:
            bq_client = get_bigquery_client()
            
            if bq_client:
                # Get detailed status
                detailed_status = bq_client.get_detailed_status()
                
                # Display key status indicators
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    status_icon = "✅" if detailed_status['connection_available'] else "❌"
                    st.metric("Connection Status", 
                             "Available" if detailed_status['connection_available'] else "Failed",
                             delta=None)
                
                with col2:
                    auth_method = detailed_status.get('auth_method_used', 'Unknown')
                    st.metric("Auth Method", auth_method)
                
                with col3:
                    project_id = detailed_status.get('project_id', 'Unknown')
                    st.metric("Project ID", project_id)
                
                # Display test results
                test_results = detailed_status.get('last_test_results', {})
                if test_results:
                    st.write("**Diagnostic Test Results:**")
                    
                    test_steps = [
                        ('step_1_client_available', 'Client Creation'),
                        ('step_2_simple_query', 'Query Execution'),
                        ('step_3_project_access', 'Project Access'),
                        ('step_4_dataset_access', 'Dataset Access'),
                        ('step_5_table_access', 'Table Access')
                    ]
                    
                    for step_key, step_name in test_steps:
                        if step_key in test_results:
                            result = test_results[step_key]
                            status_icon = "✅" if result else "❌"
                            st.write(f"{status_icon} **{step_name}**: {'PASS' if result else 'FAIL'}")
                
                # Display error details if any
                if detailed_status.get('connection_error_details'):
                    st.write("**Connection Error Details:**")
                    error_details = detailed_status['connection_error_details']
                    st.error(f"Step: {error_details.get('auth_step', 'Unknown')}")
                    st.error(f"Error: {error_details.get('error', 'Unknown error')}")
                
                # Display environment context
                st.write("**Environment Information:**")
                env_info = {
                    "Streamlit Available": detailed_status.get('streamlit_available', False),
                    "Streamlit Secrets Available": detailed_status.get('streamlit_secrets_available', False),
                    "Environment Variable Available": detailed_status.get('env_var_available', False),
                    "Client Exists": detailed_status.get('client_exists', False),
                    "Credentials Exist": detailed_status.get('credentials_exists', False),
                    "Timestamp": detailed_status.get('timestamp', 'Unknown')
                }
                
                for key, value in env_info.items():
                    if isinstance(value, bool):
                        icon = "✅" if value else "❌"
                        st.write(f"{icon} **{key}**: {value}")
                    else:
                        st.write(f"ℹ️ **{key}**: {value}")
                
                # Raw diagnostic data
                with st.expander("Raw Diagnostic Data (JSON)", expanded=False):
                    st.json(detailed_status)
            
            else:
                st.error("❌ BigQuery client is None - initialization failed")
                
        except Exception as e:
            st.error(f"❌ Error running diagnostics: {str(e)}")
            st.write("**Exception Details:**")
            st.code(str(e))

def display_simple_bigquery_status():
    """Display simple BigQuery status for main app"""
    try:
        bq_client = get_bigquery_client()
        
        if bq_client and bq_client.is_available():
            st.success("✅ BigQuery connection active")
            return True
        else:
            if bq_client:
                detailed_status = bq_client.get_detailed_status()
                # Get the first failed step for concise error message
                test_results = detailed_status.get('last_test_results', {})
                failed_steps = [step for step, result in test_results.items() 
                              if step.startswith('step_') and not result]
                
                if failed_steps:
                    step_name = failed_steps[0].replace('step_', '').replace('_', ' ').title()
                    st.warning(f"⚠️ BigQuery connection issue: {step_name} failed")
                else:
                    st.warning("⚠️ BigQuery connection not available")
            else:
                st.error("❌ BigQuery client initialization failed")
            return False
            
    except Exception as e:
        st.error(f"❌ BigQuery diagnostic error: {str(e)}")
        return False