import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
from io import BytesIO

# Import backend functions
from scheduler_backend import (
    prepare_data,
    distribute_audits,
    create_excel_output
)

# Page configuration
st.set_page_config(
    page_title="Audit Scheduler",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 1rem;
        color: #856404;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


def create_demand_capacity_chart(demand_capacity_df):
    """Create demand vs capacity bar chart"""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Total Demand',
        x=demand_capacity_df['Node'],
        y=demand_capacity_df['Total Demand'],
        marker_color='#FF6B6B'
    ))

    fig.add_trace(go.Bar(
        name='Yearly Capacity',
        x=demand_capacity_df['Node'],
        y=demand_capacity_df['Yearly Capacity'],
        marker_color='#4ECDC4'
    ))

    fig.update_layout(
        title='Demand vs Capacity by Node',
        xaxis_title='Node',
        yaxis_title='Audits',
        barmode='group',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_status_pie_chart(stats):
    """Create pie chart for scheduling status"""
    labels = ['Within Capacity', 'Over Capacity', 'ZFN Scheduled', 'ZFN Not Scheduled', 'Skipped']
    values = [
        stats['within_capacity'],
        stats['over_capacity'],
        stats['zfn_scheduled'],
        stats['zfn_not_scheduled'],
        stats['skipped_no_node'] + stats['skipped_zero_audits'] + stats['skipped_no_capacity']
    ]
    colors = ['#4ECDC4', '#FF6B6B', '#FFE66D', '#95A5A6', '#BDC3C7']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors
    )])

    fig.update_layout(
        title='Scheduling Status Distribution',
        height=400
    )

    return fig


def create_utilization_chart(summary_df):
    """Create utilization gauge chart"""
    fig = go.Figure()

    for idx, row in summary_df.iterrows():
        utilization = row['Utilization %']
        color = '#4ECDC4' if utilization <= 100 else '#FF6B6B'

        fig.add_trace(go.Bar(
            name=row['Node'],
            x=[row['Node']],
            y=[utilization],
            marker_color=color,
            text=[f"{utilization}%"],
            textposition='outside'
        ))

    fig.add_hline(y=100, line_dash="dash", line_color="red",
                  annotation_text="100% Capacity")

    fig.update_layout(
        title='Capacity Utilization by Node',
        xaxis_title='Node',
        yaxis_title='Utilization %',
        showlegend=False,
        height=400
    )

    return fig


def main():
    # Header
    st.markdown('<p class="main-header">📊 Audit Scheduler</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Priority-Based Audit Scheduling with Capacity Management</p>',
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Files")

        st.markdown("### Required Files")

        audit_file = st.file_uploader(
            "Upload Audit Details (CSV)",
            type=['csv'],
            help="CSV file with columns: node, class_name, risk_score, percentile_group, minimum_audit_count"
        )

        hr_metrics_file = st.file_uploader(
            "Upload HR Metrics (Excel)",
            type=['xlsx', 'xls'],
            help="Excel file with columns: Node, HC, AHT, Shrinkage"
        )

        st.markdown("---")

        st.markdown("### 📋 Required Columns")

        with st.expander("Audit Details CSV"):
            st.markdown("""
            - `node` - Node/Marketplace identifier
            - `class_name` - Class name
            - `risk_score` - Risk score (numeric)
            - `percentile_group` - Percentile group
            - `minimum_audit_count` - Minimum audits required
            """)

        with st.expander("HR Metrics Excel"):
            st.markdown("""
            - `Node` - Node/Marketplace identifier
            - `HC` - Headcount
            - `AHT` - Average Handle Time (minutes)
            - `Shrinkage` - Shrinkage percentage
            """)

        st.markdown("---")

        st.markdown("### ⚙️ About")
        st.info("""
        This tool schedules audits based on:
        - Priority scores (risk + percentile)
        - Capacity constraints
        - Gap requirements between audits
        """)

    # Main content
    if audit_file is not None and hr_metrics_file is not None:

        # Read uploaded files
        try:
            classes_data = pd.read_csv(audit_file)
            hr_metrics = pd.read_excel(hr_metrics_file)

            st.success(f"✅ Files loaded successfully!")

            # Display file info
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📄 Audit Details")
                st.info(f"**Rows:** {len(classes_data):,} | **Columns:** {len(classes_data.columns)}")
                with st.expander("Preview Data"):
                    st.dataframe(classes_data.head(10), use_container_width=True)

            with col2:
                st.markdown("### 📊 HR Metrics")
                st.info(f"**Rows:** {len(hr_metrics):,} | **Columns:** {len(hr_metrics.columns)}")
                with st.expander("Preview Data"):
                    st.dataframe(hr_metrics.head(10), use_container_width=True)

            st.markdown("---")

            # Run Scheduler Button
            if st.button("🚀 Run Audit Scheduler", type="primary", use_container_width=True):

                start_time = time.time()

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Step 1: Prepare Data
                    status_text.text("📋 Step 1/4: Preparing data...")
                    progress_bar.progress(10)

                    log_messages = []

                    def log_callback(msg):
                        log_messages.append(msg)

                    (prepared_data, yearly_capacity_dict, weekly_capacity_dict,
                     total_demand, priority_demand, backlog_demand, capacity_df) = prepare_data(
                        classes_data, hr_metrics, log_callback
                    )

                    progress_bar.progress(30)

                    # Step 2: Distribute Audits
                    status_text.text("📅 Step 2/4: Distributing audits...")

                    def progress_callback(progress, message):
                        progress_bar.progress(30 + int(progress * 40))
                        status_text.text(f"📅 Step 2/4: {message}")

                    audit_schedule, scheduling_results, weekly_loads, stats = distribute_audits(
                        prepared_data, yearly_capacity_dict, weekly_capacity_dict,
                        total_demand, priority_demand, backlog_demand, progress_callback
                    )

                    progress_bar.progress(70)

                    # Step 3: Create Excel Output
                    status_text.text("📝 Step 3/4: Creating Excel output...")

                    excel_output, summary_df, detailed_df, demand_capacity_df = create_excel_output(
                        audit_schedule, scheduling_results,
                        yearly_capacity_dict, weekly_capacity_dict,
                        total_demand, priority_demand, backlog_demand,
                        prepared_data, weekly_loads
                    )

                    progress_bar.progress(90)

                    # Step 4: Display Results
                    status_text.text("📊 Step 4/4: Generating visualizations...")

                    elapsed_time = time.time() - start_time

                    progress_bar.progress(100)
                    status_text.text(f"✅ Completed in {elapsed_time:.2f} seconds!")

                    st.success(f"🎉 Scheduling completed successfully in {elapsed_time:.2f} seconds!")

                    # Store results in session state
                    st.session_state['results'] = {
                        'excel_output': excel_output,
                        'summary_df': summary_df,
                        'detailed_df': detailed_df,
                        'demand_capacity_df': demand_capacity_df,
                        'stats': stats,
                        'capacity_df': capacity_df
                    }

                except Exception as e:
                    st.error(f"❌ Error during scheduling: {str(e)}")
                    import traceback
                    with st.expander("Show Error Details"):
                        st.code(traceback.format_exc())

            # Display Results if available
            if 'results' in st.session_state:
                results = st.session_state['results']

                st.markdown("---")
                st.markdown("## 📊 Results")

                # Download Button
                st.download_button(
                    label="📥 Download Excel Report",
                    data=results['excel_output'],
                    file_name=f"audit_schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )

                # Key Metrics
                st.markdown("### 📈 Key Metrics")

                stats = results['stats']

                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    total_scheduled = stats['within_capacity'] + stats['over_capacity'] + stats['zfn_scheduled']
                    st.metric("Total Scheduled", f"{total_scheduled:,}")

                with col2:
                    st.metric("Within Capacity", f"{stats['within_capacity']:,}",
                              delta=None)

                with col3:
                    st.metric("Over Capacity", f"{stats['over_capacity']:,}",
                              delta=f"-{stats['over_capacity']}" if stats['over_capacity'] > 0 else None,
                              delta_color="inverse")

                with col4:
                    st.metric("ZFN Scheduled", f"{stats['zfn_scheduled']:,}")

                with col5:
                    total_skipped = stats['skipped_no_node'] + stats['skipped_zero_audits'] + stats[
                        'skipped_no_capacity']
                    st.metric("Skipped", f"{total_skipped:,}")

                # Charts
                st.markdown("### 📊 Visualizations")

                tab1, tab2, tab3 = st.tabs(["Demand vs Capacity", "Status Distribution", "Utilization"])

                with tab1:
                    fig = create_demand_capacity_chart(results['demand_capacity_df'])
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    fig = create_status_pie_chart(stats)
                    st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    fig = create_utilization_chart(results['summary_df'])
                    st.plotly_chart(fig, use_container_width=True)

                # Data Tables
                st.markdown("### 📋 Data Tables")

                tab1, tab2, tab3, tab4 = st.tabs([
                    "Summary by Node",
                    "Demand vs Capacity",
                    "Capacity Details",
                    "Detailed Results"
                ])

                with tab1:
                    st.dataframe(results['summary_df'], use_container_width=True, height=400)

                with tab2:
                    st.dataframe(results['demand_capacity_df'], use_container_width=True, height=400)

                with tab3:
                    st.dataframe(results['capacity_df'], use_container_width=True, height=400)

                with tab4:
                    # Add filter for detailed results
                    status_filter = st.multiselect(
                        "Filter by Status",
                        options=results['detailed_df']['Status'].unique().tolist(),
                        default=results['detailed_df']['Status'].unique().tolist()
                    )
                    filtered_df = results['detailed_df'][
                        results['detailed_df']['Status'].isin(status_filter)
                    ]
                    st.dataframe(filtered_df, use_container_width=True, height=400)

                # Skipped Classes Summary
                if stats['skipped_no_node'] > 0 or stats['skipped_zero_audits'] > 0 or stats[
                    'skipped_no_capacity'] > 0:
                    st.markdown("### ⚠️ Skipped Classes")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.warning(f"**No Node Assigned:** {stats['skipped_no_node']:,}")

                    with col2:
                        st.warning(f"**Zero Audits Required:** {stats['skipped_zero_audits']:,}")

                    with col3:
                        st.warning(f"**No Capacity:** {stats['skipped_no_capacity']:,}")

        except Exception as e:
            st.error(f"❌ Error reading files: {str(e)}")
            with st.expander("Show Error Details"):
                import traceback
                st.code(traceback.format_exc())

    else:
        # Show instructions when no files uploaded
        st.info("👈 Please upload both required files in the sidebar to get started.")

        # Sample data format
        st.markdown("### 📝 Sample Data Format")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Audit Details CSV")
            sample_audit = pd.DataFrame({
                'node': ['Node_A', 'Node_A', 'Node_B'],
                'class_name': ['Class_1', 'Class_2', 'Class_3'],
                'risk_score': [10.2, 8.5, 6.0],
                'percentile_group': ['80th Percentile', '50th Percentile', '0 FN'],
                'minimum_audit_count': [52, 12, 4]
            })
            st.dataframe(sample_audit, use_container_width=True)

        with col2:
            st.markdown("#### HR Metrics Excel")
            sample_hr = pd.DataFrame({
                'Node': ['Node_A', 'Node_B'],
                'HC': [10, 5],
                'AHT': [30, 45],
                'Shrinkage': [0.8, 0.75]
            })
            st.dataframe(sample_hr, use_container_width=True)

        # Feature highlights
        st.markdown("### ✨ Features")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **🎯 Priority Scheduling**
            - Risk score-based prioritization
            - Percentile group consideration
            - RS 10.2 classes get highest priority
            """)

        with col2:
            st.markdown("""
            **📊 Capacity Management**
            - Weekly capacity constraints
            - Over-capacity handling
            - ZFN class scheduling
            """)

        with col3:
            st.markdown("""
            **📈 Gap-Based Scheduling**
            - Maintains required gaps
            - Weekly to yearly patterns
            - Even distribution
            """)


if __name__ == "__main__":
    main()