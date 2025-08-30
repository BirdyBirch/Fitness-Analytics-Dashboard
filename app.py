import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import os

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Fitness Analytics Dashboard",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional styling with Material Design icons
def apply_professional_styling():
    st.markdown("""
    <style>
        /* Import Google Fonts and Material Icons */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/icon?family=Material+Icons+Outlined');
        
        /* Professional styling */
        .chart-container {
            background: #181920;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
        }
        
        .chart-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e2e8f0;
        }
        
        /* Material Icon styling */
        .material-icons-outlined {
            font-family: 'Material Icons Outlined';
            font-weight: normal;
            font-style: normal;
            font-size: 24px;
            display: inline-block;
            line-height: 1;
            text-transform: none;
            letter-spacing: normal;
            word-wrap: normal;
            white-space: nowrap;
            direction: ltr;
            vertical-align: middle;
        }
        
        /* Professional KPI cards */
        .kpi-card {
            background: #181920;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.18);
            border-left: 4px solid;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.22);
        }
        .kpi-card.pink { border-left-color: #FF69B4; }
        .kpi-card.lilac { border-left-color: #BA55D3; }
        .kpi-card.blue { border-left-color: #4169E1; }
        .kpi-header {
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }
        .kpi-icon {
            width: 40px;
            height: 40px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 12px;
        }
        .kpi-icon.pink {
            background: rgba(255, 105, 180, 0.18);
            color: #FF69B4;
        }
        .kpi-icon.lilac {
            background: rgba(186, 85, 211, 0.18);
            color: #BA55D3;
        }
        .kpi-icon.blue {
            background: rgba(65, 105, 225, 0.18);
            color: #4169E1;
        }
        .kpi-title {
            font-size: 0.875rem;
            color: #b0b6c2;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .kpi-value {
            font-size: 2.25rem;
            font-weight: 700;
            color: #fff;
            line-height: 1;
            margin-bottom: 0.5rem;
        }
        .kpi-delta {
            font-size: 0.875rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            color: #fff;
        }
        .kpi-delta.positive { color: #48bb78; }
        .kpi-delta.negative { color: #f56565; }
        .kpi-delta-icon {
            font-size: 16px;
            margin-right: 4px;
        }
        .sync-date {
            font-size: 0.75rem;
            color: #b0b6c2;
            margin-top: 0.5rem;
            font-style: italic;
        }
        
        /* Main title gradient */
        h1 {
            background: linear-gradient(90deg, #FF69B4, #BA55D3, #4169E1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.5rem;
            font-weight: 800;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

apply_professional_styling()


# Professional KPI card with Material Icons
def create_kpi_card(title, value, delta, color_theme="pink", icon_name="monitoring", sync_date=None):
    """Create a professional KPI card with Material Design icons and optional sync date"""
    
    delta_class = "positive" if "+" in str(delta) else "negative" if "-" in str(delta) else ""
    delta_icon = "trending_up" if "+" in str(delta) else "trending_down" if "-" in str(delta) else "trending_flat"
    
    # Format sync date if provided
    sync_info = ""
    if sync_date:
        if isinstance(sync_date, pd.Timestamp):
            formatted_date = sync_date.strftime("%d.%m.%Y")
        else:
            formatted_date = str(sync_date)[:10]  # Take first 10 characters for date
        sync_info = f'<div class="sync-date">Last sync: {formatted_date}</div>'
    
    html = f"""
    <div class="kpi-card {color_theme}">
        <div class="kpi-header">
            <div class="kpi-icon {color_theme}">
                <span class="material-icons-outlined">{icon_name}</span>
            </div>
            <div class="kpi-title">{title}</div>
        </div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-delta {delta_class}">
            <span class="material-icons-outlined kpi-delta-icon">{delta_icon}</span>
            {delta}
        </div>
        {sync_info}
    </div>
    """
    return html


@st.cache_data
def load_data():
    """Load sample data for demonstration with realistic 2025 date range"""
    np.random.seed(42)
    # June 1, 2025 to August 18, 2025 (your actual data collection period)
    dates = pd.date_range(start='2025-06-01', end='2025-08-18', freq='D')
    
    # Generate realistic patterns based on your actual Renpho data (54.8-57kg range)
    weight_base = 56.2  # Realistic baseline from your actual data
    weight_trend = -0.003  # Very gradual weight loss trend observed in your data
    weight = []
    for i in range(len(dates)):
        seasonal = 0.2 * np.sin(2 * np.pi * i / 90)  # Summer variation
        weekly = 0.3 * np.sin(2 * np.pi * i / 7)  # Weekly variation
        noise = np.random.randn() * 0.35  # Daily variation similar to your data
        weight.append(max(54.5, min(57.5, weight_base + weight_trend * i + seasonal + weekly + noise)))
    
    # Steps with realistic patterns based on your activity data
    steps = []
    activities_per_week = []  # Track weekly activity types
    
    for i, date in enumerate(dates):
        base = 6000  # Lower baseline for your lifestyle
        
        # Weekly activity pattern based on your actual data
        week_day = date.weekday()
        if week_day in [0, 2, 4]:  # Mon, Wed, Fri - workout days
            base = 8500  # Higher on workout days
        elif week_day in [5, 6]:  # Weekend
            base = 5500  # Lower on weekends
        
        # Add some high-activity days (long runs, hiking)
        if np.random.random() < 0.1:  # 10% chance of high activity
            base = 12000  # Days with hiking or long runs
            
        daily_var = np.random.normal(0, 1200)
        steps.append(max(2000, base + daily_var))
        
        # Track activity types for additional visualizations
        if i % 7 < 3:  # 3 workouts per week average
            activity_type = np.random.choice(['Indoor Cycling', 'Running', 'Strength Training', 'Yoga'], 
                                           p=[0.4, 0.25, 0.25, 0.1])
            activities_per_week.append(activity_type)
    
    # Sleep patterns based on your actual Garmin data (many gaps, 5.5-15h range)
    sleep_hours = []
    for i, date in enumerate(dates):
        # Add some missing days like in your real data (~30% missing)
        if np.random.random() < 0.3:
            sleep_hours.append(np.nan)  # Missing data
        else:
            # Real sleep range from your data: 5h 35min to 15h 45min, average ~8.5h
            base = 8.5
            if date.weekday() in [4, 5]:  # Friday, Saturday - longer sleep
                base = 9.5
            elif date.weekday() == 0:  # Monday - shorter sleep
                base = 7.5
            # Add realistic variation (5.5 to 11 hours mostly)
            sleep_val = max(5.5, min(11.0, np.random.normal(base, 1.2)))
            sleep_hours.append(sleep_val)
    
    # Ensure the last recorded sleep matches your actual data: 7h 22min from Aug 16
    # Find the last non-NaN entry and set it to 7.37 hours (7h 22min)
    last_valid_idx = len(sleep_hours) - 1
    while last_valid_idx >= 0 and pd.isna(sleep_hours[last_valid_idx]):
        last_valid_idx -= 1
    if last_valid_idx >= 0:
        sleep_hours[last_valid_idx] = 7.37  # 7h 22min = 7.37 hours
    
    # Heart rate with daily patterns
    heart_rate = 65 + 5 * np.sin(2 * np.pi * np.arange(len(dates)) / len(dates)) + np.random.randn(len(dates)) * 3
    
    # Body composition based on your actual Renpho data ranges
    body_fat = 21.0 - np.arange(len(dates)) * 0.002 + np.random.randn(len(dates)) * 0.15  # 20.3-21.9%
    muscle_mass = 41.6 + np.arange(len(dates)) * 0.001 + np.random.randn(len(dates)) * 0.1  # ~41-42kg
    
    # Calories and activity adjusted for your actual weight range
    calories = [s * 0.25 + 1200 + np.random.randint(-150, 150) for s in steps]  # Lower baseline for your size
    active_minutes = [s / 100 + np.random.randint(10, 30) for s in steps]
    
    return pd.DataFrame({
        'date': dates,
        'weight': weight,
        'steps': steps,
        'calories': calories,
        'sleep_hours': sleep_hours,
        'heart_rate': heart_rate,
        'body_fat': body_fat,
        'muscle_mass': muscle_mass,
        'active_minutes': active_minutes,
        'vo2_max': 42 + np.arange(len(dates)) * 0.015 + np.random.randn(len(dates)) * 1.2  # Better for your size
    })

# Load data
df = load_data()

# Main Dashboard Title
st.markdown("<h1>💪 Fitness Analytics Dashboard</h1>", unsafe_allow_html=True)

# Core Data KPI Cards Section
st.markdown("## 📊 Core Health Metrics")

# Create KPI cards with latest data
if df is not None and not df.empty:
    # Get latest data
    latest_data = df.iloc[-1]
    latest_date = latest_data['date']
    
    # Calculate trends (last 5 days vs previous 5 days for shorter dataset)
    if len(df) >= 10:
        recent_days = df.iloc[-5:]
        previous_days = df.iloc[-10:-5]
    else:
        # Fallback for very short datasets
        recent_days = df.iloc[-3:] if len(df) >= 3 else df
        previous_days = df.iloc[-6:-3] if len(df) >= 6 else df.iloc[:-3] if len(df) > 3 else df
    
    # KPI calculations with updated variable names
    current_weight = latest_data['weight']
    weight_change = recent_days['weight'].mean() - previous_days['weight'].mean()
    weight_delta = f"{weight_change:+.1f} kg" if weight_change != 0 else "No change"
    
    current_sleep = latest_data['sleep_hours']
    sleep_avg = recent_days['sleep_hours'].mean()
    sleep_change = sleep_avg - previous_days['sleep_hours'].mean()
    sleep_delta = f"{sleep_change:+.1f}h avg" if sleep_change != 0 else "Stable"
    
    current_steps = latest_data['steps']
    steps_avg = recent_days['steps'].mean()
    steps_change = steps_avg - previous_days['steps'].mean()
    steps_delta = f"{steps_change:+.0f} daily avg" if steps_change != 0 else "Consistent"
    
    current_heart_rate = latest_data['heart_rate']
    hr_avg = recent_days['heart_rate'].mean()
    hr_change = hr_avg - previous_days['heart_rate'].mean()
    hr_delta = f"{hr_change:+.1f} BPM avg" if hr_change != 0 else "Stable"
    
    # Display KPI cards in columns
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.markdown(create_kpi_card(
            title="Last Recorded Weight",
            value=f"{current_weight:.1f} kg",
            delta=weight_delta,
            color_theme="pink",
            icon_name="scale",
            sync_date=latest_date
        ), unsafe_allow_html=True)
    
    with kpi_col2:
        # Use August 16th as the actual last sleep record date
        sleep_sync_date = pd.to_datetime('2025-08-16').strftime('%b %d')
        st.markdown(create_kpi_card(
            title="Last Recorded Sleep",
            value="7.4h",  # 7h 22min from your actual Garmin data
            delta=sleep_delta,
            color_theme="lilac",
            icon_name="bedtime",
            sync_date=sleep_sync_date
        ), unsafe_allow_html=True)
    
    with kpi_col3:
        st.markdown(create_kpi_card(
            title="Daily Steps",
            value=f"{current_steps:,.0f}",
            delta=steps_delta,
            color_theme="blue",
            icon_name="directions_walk",
            sync_date=latest_date
        ), unsafe_allow_html=True)
    
    with kpi_col4:
        st.markdown(create_kpi_card(
            title="Resting Heart Rate",
            value=f"{current_heart_rate:.0f} BPM",
            delta=hr_delta,
            color_theme="pink",
            icon_name="favorite",
            sync_date=latest_date
        ), unsafe_allow_html=True)

st.markdown("## 📊 Detailed Analytics & Trends")

if df is not None:
    # Body Composition Analysis - Full Width Section
    # Removed empty white section above Health Metrics Overview
    
    # Create two columns for bottom charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Step Count Chart (renamed from Activity Overview)
        st.markdown("""
        <div class="chart-container" style="background: #181920;">
            <h3 class="chart-title" style="color: #fff;">🚶 Step Count</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create weekly aggregated data
        df_copy = df.copy()
        df_copy['week'] = pd.to_datetime(df_copy['date']).dt.to_period('W')
        weekly_stats = df_copy.groupby('week').agg({
            'steps': 'mean',
            'calories': 'mean', 
            'active_minutes': 'mean'
        }).tail(12)  # Last 12 weeks
        
        # Clean data - remove any NaN values
        weekly_stats = weekly_stats.dropna()
        
        # Create readable week labels like "Week 29 (2025)"
        week_labels = []
        for week_period in weekly_stats.index:
            # Convert period to timestamp to get week number
            week_start = week_period.start_time
            week_num = week_start.isocalendar()[1]  # ISO week number
            year = week_start.year
            week_labels.append(f"Week {week_num} ({year})")
        
        fig2 = go.Figure()
        
        # Weekly steps as bars with darker color
        fig2.add_trace(go.Bar(
            x=week_labels,
            y=weekly_stats['steps'].fillna(0),
            name='Avg Steps',  # Capitalized for legend
            marker_color='#1A237E',  # Darker blue
            opacity=0.8
        ))
        
        # Goal line for steps
        fig2.add_hline(y=10000, line_dash="dash", line_color="#2E7D32",
                      annotation_text="Goal: 10k", 
                      annotation_position="top left",
                      annotation=dict(
                          xshift=15,  # Move annotation further right from left edge
                          yshift=8,   # Slightly higher shift
                          font=dict(size=13, color="#2E7D32", family="Inter")
                      ))
        
        fig2.update_layout(
            height=350,
            showlegend=True,
            hovermode='x unified',
            plot_bgcolor='#181920',
            paper_bgcolor='#181920',
            font=dict(size=14, family="Inter", color="white"),
            margin=dict(t=60, b=50, l=60, r=60),
            xaxis_title="Week",
            yaxis_title="Steps",
            legend=dict(
                font=dict(size=11, color="white"),
                bgcolor="rgba(24,25,32,0.95)",
                bordercolor="#333",
                borderwidth=1,
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                entrywidth=70,
                entrywidthmode="pixels",
                itemsizing="constant"
            )
        )
        # Update axis labels to white
        fig2.update_xaxes(title_font=dict(size=14, color="white"), tickfont=dict(color="white"), gridcolor='rgba(255,255,255,0.08)')
        fig2.update_yaxes(title_font=dict(size=14, color="white"), tickfont=dict(color="white"), gridcolor='rgba(255,255,255,0.08)')
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Add Activity Analysis Chart based on your Garmin data
        
        # Create activity type distribution based on your actual Garmin data
        activity_types = ['Indoor Cycling', 'Running', 'Strength Training', 'Walking/Hiking', 'Yoga']
        activity_counts = [12, 6, 3, 2, 3]  # Based on your actual data
        activity_colors = ['#7B1FA2', '#E91E63', '#F57C00', '#2E7D32', '#1976D2']
        
        # Create pie chart for activity distribution
# Add Activity Analysis Chart based on your Garmin data
with chart_col2:
    st.markdown("""
    <div class="chart-container">
        <h3 class="chart-title" style="color: #fff;">🏃‍♀️ Activity Type Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create activity type distribution based on your actual Garmin data
    activity_types = ['Indoor Cycling', 'Running', 'Strength', 'Hiking  ', 'Yoga']
    activity_counts = [12, 6, 3, 2, 3]  # Based on your actual data
    activity_colors = ['#7B1FA2', '#E91E63', '#F57C00', '#2E7D32', '#1976D2']
    
    # Create pie chart for activity distribution
    fig_activities = go.Figure(data=[go.Pie(
        labels=activity_types,
        values=activity_counts,
        hole=0.4,
        marker_colors=activity_colors,
        textinfo='percent',  # Show only percent in chart
        textfont=dict(size=13, color='white'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        # Removed domain to use full width
    )])
    fig_activities.update_layout(
        height=350,
        showlegend=True,
        plot_bgcolor='#181920',
        paper_bgcolor='#181920',
        font=dict(size=12, family="Inter", color="white"),
        margin=dict(t=40, b=70, l=40, r=40),
        legend=dict(
            orientation="h",
            yanchor="top",  # Nur dieser bleibt
            y=-0.15,
            xanchor="center",
            x=0.5,
            # yanchor='middle',  # Entferne diese Zeile
            bgcolor='rgba(24,25,32,0.95)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            borderwidth=1,
            font=dict(
                size=11,
                color='white'
            ),
            itemsizing='constant',
            itemwidth=30,
            tracegroupgap=1,
            itemclick=False,
            itemdoubleclick=False
        )
    )
    st.plotly_chart(fig_activities, use_container_width=True)
# Add Activity Timeline Chart
st.markdown("""
<div class="chart-container" style="background: #181920;">
    <h3 class="chart-title" style="color: #fff;">🏋️‍♀️ Workout Intensity & Performance Timeline</h3>
</div>
""", unsafe_allow_html=True)

# Create timeline chart based on your actual Garmin activities
activity_dates = pd.date_range(start='2025-06-27', periods=25, freq='2D')  # Every 2 days roughly
workout_types = ['Indoor Cycling', 'Running', 'Strength Training', 'Hiking', 'Yoga']

# Simulate workout data based on your actual activities
workout_data = []
colors_map = {'Indoor Cycling': '#7B1FA2', 'Running': '#E91E63', 'Strength Training': '#F57C00', 
              'Hiking': '#2E7D32', 'Yoga': '#1976D2'}

for i, date in enumerate(activity_dates):
    workout_type = np.random.choice(workout_types, p=[0.45, 0.25, 0.15, 0.1, 0.05])
    
    if workout_type == 'Indoor Cycling':
        duration = np.random.uniform(25, 65)  # 25-65 minutes
        calories = duration * 7  # ~7 cal/min
    elif workout_type == 'Running':
        duration = np.random.uniform(20, 56)  # 20-56 minutes
        calories = duration * 8.5  # ~8.5 cal/min
    elif workout_type == 'Strength Training':
        duration = np.random.uniform(40, 50)  # 40-50 minutes
        calories = duration * 5  # ~5 cal/min
    elif workout_type == 'Hiking':
        duration = np.random.uniform(60, 135)  # 60-135 minutes
        calories = duration * 4.5  # ~4.5 cal/min
    else:  # Yoga
        duration = np.random.uniform(20, 25)  # 20-25 minutes
        calories = duration * 2.5  # ~2.5 cal/min
    
    workout_data.append({
        'date': date,
        'type': workout_type,
        'duration': duration,
        'calories': calories,
        'color': colors_map[workout_type]
    })

workout_df = pd.DataFrame(workout_data)

# Create dual-axis chart
fig_timeline = make_subplots(specs=[[{"secondary_y": True}]])

# Add workout duration bars
# Add a separate bar trace for each activity type
for activity_type, color in colors_map.items():
    activity_mask = workout_df['type'] == activity_type
    fig_timeline.add_trace(
        go.Bar(
            x=workout_df.loc[activity_mask, 'date'],
            y=workout_df.loc[activity_mask, 'duration'],
            name=activity_type,
            marker_color=color,
            opacity=0.7,
            hovertemplate=f'<b>{activity_type}</b><br>Duration: %{{y:.0f}} min<br>Date: %{{x}}<extra></extra>'
        ),
        secondary_y=False
    )

# Add calories line (yellow for visibility)
fig_timeline.add_trace(
    go.Scatter(
        x=workout_df['date'],
        y=workout_df['calories'],
        name='Calories Burned',
        line=dict(color='#FFD600', width=3),  # Bright yellow
        mode='lines+markers',
        marker=dict(size=6, color='#FFD600'),
        hovertemplate='<b>Calories</b><br>Burned: %{y:.0f} cal<br>Date: %{x}<extra></extra>'
    ),
    secondary_y=True
)

# Update layout
fig_timeline.update_layout(
    height=400,
    hovermode='x unified',
    plot_bgcolor='#181920',
    paper_bgcolor='#181920',
    font=dict(size=12, family="Inter", color="white"),
    margin=dict(t=40, b=60, l=60, r=60),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=11, color="white"),
        bgcolor="rgba(24,25,32,0.95)",
        bordercolor="#333",
        borderwidth=1,
        entrywidth=90,
        entrywidthmode="pixels",
        itemsizing="constant"
    )
)
# Update axis labels to white
fig_timeline.update_xaxes(title_font=dict(size=12, color="white"), tickfont=dict(color="white"), gridcolor='rgba(255,255,255,0.08)')
fig_timeline.update_yaxes(title_font=dict(size=12, color="white"), tickfont=dict(color="white"), gridcolor='rgba(255,255,255,0.08)')


# Update axes
fig_timeline.update_xaxes(
    title_text="Date",
    title_font=dict(size=12, color="white"),
    tickfont=dict(color="white")
)
fig_timeline.update_yaxes(
    title_text="Duration (minutes)",
    secondary_y=False,
    title_font=dict(size=12, color="white"),
    tickfont=dict(color="white")
)
fig_timeline.update_yaxes(
    title_text="Calories Burned",
    secondary_y=True,
    title_font=dict(size=12, color="white"),
    tickfont=dict(color="white")
)

# Display the Workout Intensity & Performance Timeline chart
st.plotly_chart(fig_timeline, use_container_width=True)

## Full width advanced charts with fixed tabs

# Single tab definition
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Health Metrics Overview",
    "💤 Sleep Patterns", 
    "🎯 Goal Progress",
    "📈 Correlations"
])

with tab1:
    # Create comprehensive health dashboard with better spacing
    fig3 = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Heart Rate Trend", "VO2 Max", 
                      "Steps Distribution", "Sleep Quality"),
        vertical_spacing=0.25,
        horizontal_spacing=0.18,
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "scatter"}]]
    )
    # Heart rate over time with darker color
    fig3.add_trace(
        go.Scatter(
            x=df['date'].iloc[-30:],
            y=df['heart_rate'].iloc[-30:],
            mode='lines',
            name='Heart Rate',
            line=dict(color='#C62828', width=3)
        ),
        row=1, col=1
    )
    # VO2 Max with proper data handling
    if 'vo2_max' in df.columns and not df['vo2_max'].dropna().empty:
        vo2_data = df[df['vo2_max'].notna()]
        fig3.add_trace(
            go.Scatter(
                x=vo2_data['date'].iloc[-90:],
                y=vo2_data['vo2_max'].iloc[-90:],
                mode='markers+lines',
                name='VO2 Max',
                line=dict(color='#1565C0', width=3),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
    else:
        sample_vo2 = 45 + np.random.randn(30) * 2
        fig3.add_trace(
            go.Scatter(
                x=df['date'].iloc[-30:],
                y=sample_vo2,
                mode='lines',
                name='VO2 Max (estimated)',
                line=dict(color='#1565C0', width=3, dash='dash'),
                opacity=0.7
            ),
            row=1, col=2
        )
    # Steps histogram with darker color
    if 'steps' in df.columns and not df['steps'].dropna().empty:
        fig3.add_trace(
            go.Histogram(
                x=df['steps'].iloc[-90:].dropna(),
                name='Steps Distribution',
                marker_color='#6A1B9A',
                nbinsx=25,
                opacity=0.8
            ),
            row=2, col=1
        )
    # Sleep quality scatter
    if 'sleep_hours' in df.columns and not df['sleep_hours'].dropna().empty:
        sleep_data = df[df['sleep_hours'].notna()]
        sleep_quality = (sleep_data['sleep_hours'] / 8 * 100).iloc[-30:]
        fig3.add_trace(
            go.Scatter(
                x=sleep_data['date'].iloc[-30:],
                y=sleep_quality,
                mode='markers',
                name='Sleep Quality',
                marker=dict(
                    size=12,
                    color=sleep_quality,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=dict(text="Quality %", font=dict(size=12, color="white")))
                )
            ),
            row=2, col=2
        )
    else:
        fig3.add_trace(
            go.Scatter(
                x=[df['date'].iloc[-1]],
                y=[50],
                mode='markers+text',
                name='No Sleep Data',
                marker=dict(size=15, color='gray'),
                text=["No Data Available"],
                textposition="middle center",
                textfont=dict(size=12)
            ),
            row=2, col=2
        )
    fig3.update_layout(
        height=600,
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='#181920',
        paper_bgcolor='#181920',
        font=dict(size=12, family="Inter", color="white"),
        margin=dict(t=100, b=60, l=80, r=60),
        title=dict(
            text="<b>Health Metrics Overview</b>",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=20, color="white")
        )
    )
    fig3.update_annotations(
        font=dict(size=14, color="white", family="Inter"),
        yshift=10
    )
    fig3.update_xaxes(
        title_font=dict(size=11, color="white"),
        tickfont=dict(size=10, color="white"),
        showgrid=True,
        gridcolor='rgba(255,255,255,0.08)'
    )
    fig3.update_yaxes(
        title_font=dict(size=11, color="white"),
        tickfont=dict(size=10, color="white"),
        showgrid=True,
        gridcolor='rgba(255,255,255,0.08)'
    )
    fig3.update_xaxes(title_text="Date", row=1, col=1)
    fig3.update_yaxes(title_text="BPM", row=1, col=1)
    fig3.update_xaxes(title_text="Date", row=1, col=2)
    fig3.update_yaxes(title_text="ml/kg/min", row=1, col=2)
    fig3.update_xaxes(title_text="Daily Steps", row=2, col=1)
    fig3.update_yaxes(title_text="Frequency", row=2, col=1)
    fig3.update_xaxes(title_text="Date", row=2, col=2)
    fig3.update_yaxes(title_text="Quality %", row=2, col=2)
    st.plotly_chart(fig3, use_container_width=True)
    
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        
        if 'sleep_hours' in df.columns and not df['sleep_hours'].dropna().empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Sleep pattern heatmap with better data handling
                sleep_data = df[df['sleep_hours'].notna()].copy()
                sleep_data['day_name'] = pd.to_datetime(sleep_data['date']).dt.day_name()
                sleep_data['month_name'] = pd.to_datetime(sleep_data['date']).dt.month_name()
                
                sleep_matrix = sleep_data.pivot_table(
                    values='sleep_hours',
                    index='day_name',
                    columns='month_name',
                    aggfunc='mean'
                )
                
                if not sleep_matrix.empty:
                    fig_sleep = go.Figure(data=go.Heatmap(
                        z=sleep_matrix.values,
                        x=sleep_matrix.columns,
                        y=sleep_matrix.index,
                        colorscale='Blues',
                        text=np.round(sleep_matrix.values, 1),
                        texttemplate='%{text}h',
                        textfont={"size": 10, "color": "black"},
                        colorbar=dict(
                            title=dict(text="Hours", font=dict(color="white")),
                            tickfont=dict(color="white")
                        )
                    ))
                    
                    fig_sleep.update_layout(
                        title={
                            'text': "Sleep Pattern Heatmap (Average Hours by Day/Month)",
                            'font': dict(size=16, color="white")
                        },
                        height=400,
                        xaxis_title="Month",
                        yaxis_title="Day of Week",
                        font=dict(size=14, color="white"),
                        plot_bgcolor='#181920',
                        paper_bgcolor='#181920'
                    )
                    st.plotly_chart(fig_sleep, use_container_width=True)
                else:
                    st.info("Not enough sleep data for heatmap visualization")
            
            with col2:
                st.markdown("### 📊 Sleep Statistics")
                sleep_data = df['sleep_hours'].dropna()
                
                if not sleep_data.empty:
                    avg_sleep = sleep_data.mean()
                    best_night = sleep_data.max()
                    worst_night = sleep_data.min()
                    
                    st.metric("Average Sleep", f"{avg_sleep:.1f} hrs")
                    st.metric("Best Night", f"{best_night:.1f} hrs")
                    st.metric("Worst Night", f"{worst_night:.1f} hrs")
                    
                    # Sleep recommendation
                    if avg_sleep < 7:
                        st.warning("⚠️ Below recommended 7-9 hours")
                    elif avg_sleep > 9:
                        st.info("ℹ️ Above typical range")
                    else:
                        st.success("✅ Within healthy range")
                else:
                    st.info("No sleep data available for statistics")
        else:
            st.info("🛌 No sleep data available to display sleep analysis.")
    
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Goal tracking with safe data handling
        goals = {}
        
        # Only add goals if data exists
        if 'weight' in df.columns and not df['weight'].dropna().empty:
            current_weight = df['weight'].dropna().iloc[-1]
            goals['Weight'] = {'current': current_weight, 'goal': 55, 'unit': 'kg'}
        
        if 'body_fat' in df.columns and not df['body_fat'].dropna().empty:
            current_bf = df['body_fat'].dropna().iloc[-1]
            goals['Body Fat'] = {'current': current_bf, 'goal': 20, 'unit': '%'}
        
        if 'steps' in df.columns and not df['steps'].dropna().empty:
            avg_steps = df['steps'].dropna().iloc[-7:].mean()
            goals['Daily Steps'] = {'current': avg_steps, 'goal': 10000, 'unit': 'steps'}
        
        if 'sleep_hours' in df.columns and not df['sleep_hours'].dropna().empty:
            avg_sleep = df['sleep_hours'].dropna().iloc[-7:].mean()
            goals['Sleep'] = {'current': avg_sleep, 'goal': 8, 'unit': 'hrs'}
        
        if goals:  # Only create chart if we have goals
            fig_goals = go.Figure()
            colors = ['#E91E63', '#F57C00', '#7B1FA2', '#1565C0']
            for i, (metric, values) in enumerate(goals.items()):
                progress = (values['current'] / values['goal']) * 100
                if metric in ['Weight', 'Body Fat']:
                    progress = 200 - progress  # Inverse for reduction goals
                fig_goals.add_trace(go.Bar(
                    x=[min(progress, 150)],  # Cap at 150% for display
                    y=[metric],
                    orientation='h',
                    name=metric,
                    marker_color=colors[i % len(colors)],
                    text=f"{values['current']:.1f} / {values['goal']} {values['unit']}",
                    textposition='inside',
                    textfont=dict(size=14, color='black'),
                    showlegend=False
                ))
            fig_goals.update_layout(
                title={
                    'text': "Goal Progress Tracker",
                    'font': dict(size=16, color="white")
                },
                xaxis_title="Progress (%)",
                yaxis_title="",
                height=400,
                xaxis=dict(range=[0, 150], title_font=dict(color="white"), tickfont=dict(color="white")),
                yaxis=dict(tickfont=dict(color="white")),
                bargap=0.3,
                plot_bgcolor='#181920',
                paper_bgcolor='#181920',
                font=dict(size=14, color="white")
            )
            # Add goal line with annotation above the top bar
            fig_goals.add_vline(
                x=100,
                line_dash="dash",
                line_color="#2E7D32",
                annotation_text="Goal",
                annotation_font_color="#2E7D32",
                annotation_position="top right",
                annotation_y=1.02,
                annotation_yref="paper",
                annotation_yanchor="bottom"
            )
            st.plotly_chart(fig_goals, use_container_width=True)
        else:
            st.info("No data available to track goals. Please ensure your fitness data is loaded.")
    
    with tab4:
        st.markdown("<br>", unsafe_allow_html=True)
        # st.markdown("### 🔍 Health Metrics Correlations")
        
        # Create correlation matrix with available columns
        corr_cols = ['weight', 'body_fat', 'muscle_mass', 'steps', 'calories', 'sleep_hours', 'heart_rate']
        available_cols = [col for col in corr_cols if col in df.columns and not df[col].dropna().empty]
        
        if len(available_cols) >= 2:  # Need at least 2 columns for correlation
            corr_data = df[available_cols].corr()
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=corr_data.columns,
                y=corr_data.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_data.values, 2),
                texttemplate='%{text}',
                textfont={"size": 12, "color": "black"},
                colorbar=dict(
                    title=dict(text="Correlation", font=dict(size=14, color="white")),
                    tickfont=dict(color="white")
                )
            ))
            fig_corr.update_layout(
                title={
                    'text': "Correlation Matrix of Health Metrics",
                    'font': dict(size=16, color="white")
                },
                height=500,
                font=dict(size=14, color="white"),
                plot_bgcolor='#181920',
                paper_bgcolor='#181920',
                xaxis=dict(tickfont=dict(color="white")),
                yaxis=dict(tickfont=dict(color="white"))
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Key insights based on available data
            st.markdown("#### 💡 Key Insights")
            col1, col2 = st.columns(2)
            
            with col1:
                insights = []
                if 'steps' in available_cols and 'calories' in available_cols:
                    insights.append("- Steps ↔ Calories Burned")
                if 'sleep_hours' in available_cols and 'steps' in available_cols:
                    insights.append("- Sleep Quality ↔ Activity Level")
                if 'muscle_mass' in available_cols and 'vo2_max' in available_cols:
                    insights.append("- Muscle Mass ↔ VO2 Max")
                
                if insights:
                    st.info("**Positive Correlations:**\n" + "\n".join(insights))
                else:
                    st.info("**Positive Correlations:**\nAnalyzing available data...")
            
            with col2:
                inverse_insights = []
                if 'body_fat' in available_cols and 'muscle_mass' in available_cols:
                    inverse_insights.append("- Body Fat ↔ Muscle Mass")
                if 'weight' in available_cols and 'steps' in available_cols:
                    inverse_insights.append("- Weight ↔ Steps (slight)")
                if 'heart_rate' in available_cols:
                    inverse_insights.append("- Heart Rate ↔ Fitness Level")
                
                if inverse_insights:
                    st.warning("**Inverse Correlations:**\n" + "\n".join(inverse_insights))
                else:
                    st.warning("**Inverse Correlations:**\nAnalyzing available data...")
        else:
            st.info("Need at least 2 metrics with data to show correlations.")

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    
    # Date range selector
    st.markdown("#### 📅 Analysis Period")
    date_range = st.date_input(
        "Select Date Range",
        value=(datetime.now() - timedelta(days=30), datetime.now()),
        max_value=datetime.now()
    )
    
    st.markdown("### 🎯 Personal Goals")
    
    # Goal settings
    weight_goal = st.number_input("Target Weight (kg)", value=55.0, step=0.5)
    bf_goal = st.number_input("Target Body Fat (%)", value=20.0, step=0.5)
    steps_goal = st.number_input("Daily Steps Goal", value=10000, step=500)
    
    st.markdown("### 📊 Display Options")
    
    # Display preferences
    show_trends = st.checkbox("Show Trend Lines", value=True)
    show_goals = st.checkbox("Show Goal Markers", value=True)
    time_window = st.selectbox(
        "Default Time Window",
        ["7 days", "30 days", "90 days", "1 year"]
    )
    
    # Units preference
    units = st.radio("Measurement Units", ["Metric", "Imperial"])
    
    # Data management
    if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    if st.button("📥 Export Report", use_container_width=True):
        st.info("Report export feature coming soon!")
    
    st.markdown("### 📱 Connected Devices")
    st.markdown("""
    <div style='padding: 10px; background: rgba(255,255,255,0.1); border-radius: 8px;'>
        <div style='display: flex; align-items: center; margin-bottom: 8px;'>
            <span class='material-icons-outlined' style='color: #48bb78; margin-right: 8px;'>watch</span>
            <span>Garmin Watch</span>
        </div>
        <div style='display: flex; align-items: center;'>
            <span class='material-icons-outlined' style='color: #ed8936; margin-right: 8px;'>smartphone</span>
            <span>Health App</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Data preview expander
    if df is not None:
        with st.expander("📊 Data Preview"):
            st.markdown("#### Available Columns:")
            st.write(df.columns.tolist())
            st.markdown("#### Data Sample:")
            st.dataframe(df.head())
            if not df.empty:
                st.markdown(f"**Date Range:** {df['date'].min().date()} to {df['date'].max().date()}")
                st.markdown(f"**Total Records:** {len(df):,}")

# Footer with professional styling
st.markdown("""
<div style='text-align: center; padding: 2rem; background: #181920; border-radius: 15px; margin-top: 2rem;'>
    <h3 style='color: #fff; margin-bottom: 1rem;'>💪 Health & Fitness Insights</h3>
    <div style='display: flex; justify-content: space-around; flex-wrap: wrap;'>
        <div style='flex: 1; min-width: 200px; padding: 1rem;'>
            <span class='material-icons-outlined' style='color: #E91E63; font-size: 36px;'>trending_up</span>
            <p style='color: #fff; margin-top: 0.5rem;'>
                <strong style='color: #fff;'>Progress Score</strong><br>
                87% towards goals
            </p>
        </div>
        <div style='flex: 1; min-width: 200px; padding: 1rem;'>
            <span class='material-icons-outlined' style='color: #7B1FA2; font-size: 36px;'>emoji_events</span>
            <p style='color: #fff; margin-top: 0.5rem;'>
                <strong style='color: #fff;'>Current Streak</strong><br>
                12 days active
            </p>
        </div>
        <div style='flex: 1; min-width: 200px; padding: 1rem;'>
            <span class='material-icons-outlined' style='color: #1565C0; font-size: 36px;'>insights</span>
            <p style='color: #fff; margin-top: 0.5rem;'>
                <strong style='color: #fff;'>Weekly Average</strong><br>
                Above baseline
            </p>
        </div>
    </div>
        <p style='color: #a0aec0; font-size: 0.875rem; margin-top: 1.5rem;'>
        Last synced: {} | Powered by Health Analytics Engine v2.0
    </p>
</div>
""".format(datetime.now().strftime("%B %d, %Y at %I:%M %p")), unsafe_allow_html=True)
