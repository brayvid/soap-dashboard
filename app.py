# app.py
import matplotlib
matplotlib.use('Agg') # Should be very early

from dotenv import load_dotenv
load_dotenv() # Load .env before other imports that might use env vars

import os # Make sure os is imported early
from flask import Flask, render_template, request, url_for, send_from_directory, jsonify # Keep redirect for future use
import pandas as pd
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import base64
import datetime # Added for date formatting in dataset metrics

app = Flask(__name__)

# --- Database Connection ---
DEPLOY_ENV = os.environ.get('DEPLOY_ENV', 'DEVELOPMENT').upper()
# app.logger.info(f"FLASK_ENV from os.environ: {os.environ.get('FLASK_ENV')}") 
# app.logger.info(f"DEPLOY_ENV set to: {DEPLOY_ENV}")


def get_env_var(var_name_prefix, key):
    var_to_check = f"{var_name_prefix}_{key.upper()}"
    if DEPLOY_ENV == 'PRODUCTION':
        var_to_check_prod = f"{var_name_prefix}_{key.upper()}_PROD"
        val = os.environ.get(var_to_check_prod)
        return val
    else: # DEVELOPMENT or other
        val = os.environ.get(var_to_check)
        app.logger.info(f"[get_env_var DEV] Tried {var_to_check}, got: {val if val else 'None'}")
        return val

def get_engine():
    db_config = {}
    required_keys = ["username", "password", "host", "database"]
    prefix = "DB"

    for k in required_keys:
        db_config[k] = get_env_var(prefix, k)
        if db_config[k] is None:
            app.logger.error(f"CRITICAL: Missing database configuration for: {k.upper()} (DEPLOY_ENV: {DEPLOY_ENV})")
            return None
    
    db_config["port"] = get_env_var(prefix, "port") or 5432

    try:
        db_connection_str = f'postgresql+psycopg2://{db_config["username"]}:{db_config["password"]}@{db_config["host"]}:{db_config["port"]}/{db_config["database"]}'
        engine = create_engine(db_connection_str)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        app.logger.info("Database connection successful.")
        return engine
    except Exception as e:
        app.logger.error(f"Database connection failed: {e}")
        return None

engine = get_engine()

# --- Data Fetching Functions ---
def fetch_politicians_list(_engine):
    if not _engine: return pd.DataFrame({'politician_id': [], 'name': []})
    query = text("SELECT politician_id, name FROM politicians ORDER BY name ASC;")
    try:
        with _engine.connect() as connection: df = pd.read_sql(query, connection)
        return df
    except Exception as e: 
        app.logger.error(f"Error in fetch_politicians_list: {e}")
        return pd.DataFrame({'politician_id': [], 'name': []})

def fetch_sentiment_distribution_per_politician(_engine, min_total_votes_threshold=10, sort_by_total_votes=False):
    if not _engine: return pd.DataFrame()
    approve_threshold = 0.1
    disapprove_threshold = -0.1
    
    if sort_by_total_votes:
        order_by_clause_final = "ORDER BY total_votes DESC, calculated_ranking_score DESC, politician_name ASC"
    else:
        order_by_clause_final = "ORDER BY calculated_ranking_score DESC, total_votes DESC, politician_name ASC"

    query_with_final_order_by = text(f"""
        WITH VoteSentimentCategories AS (
            SELECT v.politician_id,
                CASE WHEN w.sentiment_score > {approve_threshold} THEN 'Approve'
                     WHEN w.sentiment_score < {disapprove_threshold} THEN 'Disapprove'
                     ELSE 'Neutral' END AS sentiment_category
            FROM votes AS v JOIN words AS w ON v.word_id = w.word_id WHERE w.sentiment_score IS NOT NULL 
        ), PoliticianSentimentCounts AS (
            SELECT politician_id, sentiment_category, COUNT(*) AS category_count
            FROM VoteSentimentCategories GROUP BY politician_id, sentiment_category
        ), PoliticianTotalScorableVotes AS (
            SELECT politician_id, COUNT(*) AS total_votes FROM VoteSentimentCategories GROUP BY politician_id
        ), PoliticianPercentages AS (
            SELECT p.politician_id, p.name AS politician_name, ptv.total_votes,
                COALESCE(SUM(CASE WHEN psc.sentiment_category = 'Approve' THEN psc.category_count ELSE 0 END) * 100.0 / NULLIF(ptv.total_votes, 0), 0) AS approve_percent,
                COALESCE(SUM(CASE WHEN psc.sentiment_category = 'Disapprove' THEN psc.category_count ELSE 0 END) * 100.0 / NULLIF(ptv.total_votes, 0), 0) AS disapprove_percent,
                COALESCE(SUM(CASE WHEN psc.sentiment_category = 'Neutral' THEN psc.category_count ELSE 0 END) * 100.0 / NULLIF(ptv.total_votes, 0), 0) AS neutral_percent
            FROM politicians AS p JOIN PoliticianTotalScorableVotes AS ptv ON p.politician_id = ptv.politician_id
            LEFT JOIN PoliticianSentimentCounts AS psc ON p.politician_id = psc.politician_id
            WHERE ptv.total_votes >= :min_votes_threshold 
            GROUP BY p.politician_id, p.name, ptv.total_votes
        )
        SELECT 
            politician_id, 
            politician_name,
            approve_percent,
            disapprove_percent,
            neutral_percent,
            total_votes,
            (approve_percent - disapprove_percent) AS calculated_ranking_score
        FROM PoliticianPercentages
        {order_by_clause_final};
    """)

    try:
        with _engine.connect() as connection:
            df = pd.read_sql(query_with_final_order_by, connection, params={'min_votes_threshold': min_total_votes_threshold})
        for col in ['approve_percent', 'disapprove_percent', 'neutral_percent', 'total_votes', 'calculated_ranking_score']:
            if col in df.columns: 
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except Exception as e: 
        app.logger.error(f"Error in fetch_sentiment_distribution_per_politician: {e}\nQuery was: {query_with_final_order_by}")
        return pd.DataFrame()
    
def fetch_weekly_approval_trends_for_selected_politicians(_engine, politician_ids_list):
    if not _engine or not politician_ids_list: return pd.DataFrame()
    try:
        safe_politician_ids = tuple(map(int, politician_ids_list))
    except ValueError:
        app.logger.error(f"Invalid non-integer ID in politician_ids_list for trends: {politician_ids_list}")
        return pd.DataFrame()

    if not safe_politician_ids: return pd.DataFrame()
    
    if len(safe_politician_ids) == 1:
        in_clause_sql = f"({safe_politician_ids[0]})"
    else:
        in_clause_sql = str(safe_politician_ids)

    query = text(f"""
        SELECT p.name AS politician_name, p.politician_id,           
            TO_CHAR(v.created_at, 'IYYY-IW') AS year_week,
            DATE_TRUNC('week', v.created_at)::date AS week_start_date,
            COUNT(v.vote_id) AS total_votes_in_week,
            CASE WHEN COUNT(v.vote_id) > 0 
                THEN (((SUM(w.sentiment_score) / COUNT(v.vote_id)) / 2.0) + 0.5) * 100.0 
                ELSE NULL END AS weekly_approval_rating_percent 
        FROM votes AS v JOIN words AS w ON v.word_id = w.word_id
        JOIN politicians AS p ON v.politician_id = p.politician_id 
        WHERE v.politician_id IN {in_clause_sql} 
            AND w.sentiment_score IS NOT NULL AND v.created_at IS NOT NULL
        GROUP BY p.politician_id, p.name, year_week, week_start_date 
        ORDER BY p.name ASC, week_start_date ASC;
    """)
    try:
        with _engine.connect() as connection: df = pd.read_sql(query, connection) 
        if 'weekly_approval_rating_percent' in df.columns: df['weekly_approval_rating_percent'] = pd.to_numeric(df['weekly_approval_rating_percent'], errors='coerce') 
        if 'week_start_date' in df.columns: df['week_start_date'] = pd.to_datetime(df['week_start_date'], errors='coerce')
        return df
    except Exception as e: 
        app.logger.error(f"Error in fetch_weekly_approval_trends_for_selected_politicians: {e}")
        return pd.DataFrame()

def fetch_dataset_metrics(_engine):
    if not _engine:
        return {key: "N/A" for key in ["total_politicians", "total_words_scorable", "total_votes", "votes_date_range"]}
    metrics = {}
    try:
        with _engine.connect() as connection:
            metrics["total_politicians"] = connection.execute(text("SELECT COUNT(*) FROM politicians;")).scalar_one_or_none() or "N/A"
            metrics["total_words_scorable"] = connection.execute(text("SELECT COUNT(*) FROM words WHERE sentiment_score IS NOT NULL;")).scalar_one_or_none() or "N/A"
            metrics["total_votes"] = connection.execute(text("SELECT COUNT(*) FROM votes;")).scalar_one_or_none() or "N/A"
            
            res_dates = connection.execute(text("SELECT MIN(created_at)::date AS min_date, MAX(created_at)::date AS max_date FROM votes;")).fetchone()
            if res_dates and res_dates.min_date and res_dates.max_date:
                min_d, max_d = res_dates.min_date, res_dates.max_date
                metrics["votes_date_range"] = f"{min_d.strftime('%Y-%m-%d')} to {max_d.strftime('%Y-%m-%d')}" if min_d != max_d else f"On {min_d.strftime('%Y-%m-%d')}"
            else:
                metrics["votes_date_range"] = "N/A"
        return metrics
    except Exception as e:
        app.logger.error(f"Error fetching dataset metrics: {e}")
        return {key: "Error" for key in ["total_politicians", "total_words_scorable", "total_votes", "votes_date_range"]}

def fetch_feed_updates(_engine, limit=50): # Renamed function
    if not _engine: return pd.DataFrame()
    query = text(f"""
        SELECT
            w.word AS "Word",
            p.name AS "Politician",
            v.created_at AS "Timestamp"
        FROM
            votes v
        JOIN
            words w ON v.word_id = w.word_id
        JOIN
            politicians p ON v.politician_id = p.politician_id
        ORDER BY
            v.created_at DESC
        LIMIT :limit_val;
    """)
    try:
        with _engine.connect() as connection:
            df = pd.read_sql(query, connection, params={'limit_val': limit})
        
        if not df.empty:
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            if 'Word' in df.columns:
                df['Word'] = df['Word'].astype(str).apply(lambda x: ' '.join(s.capitalize() for s in x.split()))
        return df
    except Exception as e:
        app.logger.error(f"Error fetching feed updates: {e}\nQuery: {query}")
        return pd.DataFrame()

# --- Plotting Functions ---
def plot_stacked_horizontal_bar_to_image(df, categories, category_colors, title, xlabel, ylabel, top_n=20, decimal_places=1):
    if df.empty or not all(cat in df.columns for cat in categories): return None
    data_to_plot = df.head(top_n).copy();
    for cat in categories: data_to_plot[cat] = pd.to_numeric(data_to_plot[cat], errors='coerce').fillna(0)
    plot_df_ready = data_to_plot.set_index('politician_name')[categories] 
    plot_df_ready = plot_df_ready.iloc[::-1] 
    num_politicians = len(plot_df_ready)
    fig_height = max(6, min(15, 2 + num_politicians * 0.7)); fig_width = 12
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_df_ready.plot(kind='barh', stacked=True, color=category_colors, ax=ax, width=0.8)
    ax.set_title(title, fontsize=16, pad=15, weight='bold')
    ax.set_xlabel(xlabel, fontsize=13); ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlim(0, 105); ax.legend(title="Sentiment Category", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    for p_index, (idx, row) in enumerate(plot_df_ready.iterrows()):
        cumulative_width = 0
        for i, val in enumerate(row):
            if val > 3: 
                 label_x_pos = cumulative_width + (val / 2)
                 ax.text(label_x_pos, p_index, f'{val:.{decimal_places}f}%', 
                         ha='center', va='center', color='white', fontsize=9, weight='bold',
                         bbox=dict(boxstyle="round,pad=0.1", fc='black', alpha=0.3, ec='none'))
            cumulative_width += val
    plt.tight_layout(rect=[0, 0, 0.85, 1]); img_buf = BytesIO()
    fig.savefig(img_buf, format="png", bbox_inches='tight', dpi=100); img_buf.seek(0)
    plt.close(fig); return img_buf

def plot_multiline_chart_to_image(df, x_col, y_col, group_col, title, xlabel, ylabel, color_palette="viridis", decimal_places=0):
    if df.empty or x_col not in df.columns or y_col not in df.columns or group_col not in df.columns or df[y_col].isnull().all(): return None
    unique_groups = df[group_col].nunique(); fig_height = max(6, min(12, 5 + unique_groups * 0.3)); fig_width = 14
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if unique_groups <= 10: colors = sns.color_palette("tab10", n_colors=unique_groups)
    else: colors = sns.color_palette(color_palette, n_colors=unique_groups)
    sns.lineplot(data=df, x=x_col, y=y_col, hue=group_col, marker='o', ax=ax, linewidth=2, palette=colors)
    ax.set_title(title, fontsize=16, pad=15, weight='bold'); ax.set_xlabel(xlabel, fontsize=13); ax.set_ylabel(ylabel, fontsize=13)
    plt.xticks(rotation=30, ha='right', fontsize=10); plt.yticks(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7); ax.set_ylim(-5, 105)
    for line_index, politician_name_val in enumerate(df[group_col].unique()):
        politician_df = df[df[group_col] == politician_name_val]
        for index, row in politician_df.iterrows():
            if pd.notna(row[y_col]):
                ax.text(row[x_col], row[y_col] + 2, f"{row[y_col]:.{decimal_places}f}%", 
                        color=colors[line_index % len(colors)], ha="center", va="bottom", fontsize=8, weight='bold',
                        bbox=dict(boxstyle="round,pad=0.1", fc='white', alpha=0.5, ec='none'))
    if unique_groups > 7: 
        ax.legend(title=group_col.replace('_', ' ').title(), bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize='small')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    else:
        ax.legend(title=group_col.replace('_', ' ').title(), fontsize='small'); plt.tight_layout()
    img_buf = BytesIO(); fig.savefig(img_buf, format="png", bbox_inches='tight', dpi=100); img_buf.seek(0)
    plt.close(fig); return img_buf

def plot_similarity_heatmap_to_image(similarity_matrix_df, title="Sentiment Similarity Matrix"): 
    if similarity_matrix_df.empty or len(similarity_matrix_df) < 2: return None
    plot_df = similarity_matrix_df.copy() 
    display_n = len(plot_df) 
    for col in plot_df.columns: plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    fig_height = max(8, min(25, 3 + display_n * 0.9)) 
    fig_width = fig_height * 1.2 
    if display_n < 5 : fig_height = 6; fig_width = 8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(plot_df, annot=True, fmt=".2f", cmap="viridis", linewidths=.5, ax=ax, 
                cbar=True, square=True, annot_kws={"size": 6 if display_n > 20 else (7 if display_n > 15 else 8)},
                cbar_kws={'label': 'Cosine Similarity of Sentiment Distribution', 'shrink': 0.7}) 
    ax.set_title(title, fontsize=16, pad=20, weight='bold')
    ax.tick_params(axis='x', rotation=70, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=8)
    fig.subplots_adjust(left=0.3 if display_n > 5 else 0.25, bottom=0.3 if display_n > 5 else 0.25, right=0.98, top=0.95)
    img_buf = BytesIO(); 
    fig.savefig(img_buf, format="png", bbox_inches='tight', dpi=120)
    img_buf.seek(0)
    plt.close(fig); 
    return img_buf

def get_image_as_base64(img_buf):
    if img_buf:
        img_buf.seek(0)
        return base64.b64encode(img_buf.read()).decode('utf-8')
    return None

# --- Main Dashboard Route ---
@app.route('/')
def dashboard():
    if not engine:
        return render_template('error.html', message="CRITICAL: Database connection failed. Dashboard cannot operate.")

    active_tab = request.args.get('tab', 'approval')
    politicians_list_df = fetch_politicians_list(engine)
    
    approval_data_dict = {}
    trends_data_dict = {}
    similarity_data_dict = {}
    dataset_data_dict = {}
    feed_data_dict = {} # Renamed from updates_data_dict

    if active_tab == 'approval':
        min_votes_param = request.args.get('min_votes', '10')
        min_votes = int(min_votes_param) if min_votes_param.isdigit() else 10
        df_sentiment_dist = fetch_sentiment_distribution_per_politician(
            engine, min_total_votes_threshold=min_votes, sort_by_total_votes=False
        )
        dist_img_base64 = None
        if not df_sentiment_dist.empty:
            sentiment_categories = ['approve_percent', 'neutral_percent', 'disapprove_percent']
            category_colors_map = {'approve_percent': 'mediumseagreen', 'neutral_percent': 'lightgrey', 'disapprove_percent': 'lightcoral'}
            plotted_categories = [cat for cat in sentiment_categories if cat in df_sentiment_dist.columns]
            category_colors = [category_colors_map[cat] for cat in plotted_categories if cat in category_colors_map]
            if plotted_categories:
                dist_img_buf = plot_stacked_horizontal_bar_to_image(
                    df_sentiment_dist, categories=plotted_categories, category_colors=category_colors,
                    title='Approval Rating Distribution', xlabel='Percentage of Votes (%)', ylabel=''
                )
                dist_img_base64 = get_image_as_base64(dist_img_buf)
        approval_data_dict = {
            'min_votes_current': min_votes,
            'df_sentiment_dist': df_sentiment_dist,
            'dist_img_base64': dist_img_base64
        }
    elif active_tab == 'trends':
        selected_politician_ids_str = request.args.getlist('politician_ids_trends')
        selected_politician_ids = [int(pid) for pid in selected_politician_ids_str if pid.isdigit()]

        if not selected_politician_ids_str and not politicians_list_df.empty:
            trump_row = politicians_list_df[politicians_list_df['name'].str.contains("Donald Trump", case=False, na=False)]
            if not trump_row.empty:
                selected_politician_ids = [int(trump_row['politician_id'].iloc[0])]
            elif not politicians_list_df.empty:
                selected_politician_ids = [int(politicians_list_df['politician_id'].iloc[0])]
        
        if "All" in request.args.get('politician_select_mode_trends', '') and not politicians_list_df.empty:
            selected_politician_ids_for_query = politicians_list_df['politician_id'].tolist()
        else:
            selected_politician_ids_for_query = selected_politician_ids

        weekly_df_multiple = pd.DataFrame()
        weekly_trend_img_base64 = None
        selected_politician_names = []

        if selected_politician_ids_for_query and not politicians_list_df.empty:
            selected_politician_names = politicians_list_df[politicians_list_df['politician_id'].isin(selected_politician_ids_for_query)]['name'].tolist()
            weekly_df_multiple = fetch_weekly_approval_trends_for_selected_politicians(engine, selected_politician_ids_for_query)
            if not weekly_df_multiple.empty and 'weekly_approval_rating_percent' in weekly_df_multiple.columns and weekly_df_multiple['weekly_approval_rating_percent'].notna().any():
                weekly_trend_img_buf = plot_multiline_chart_to_image(
                    weekly_df_multiple, x_col='week_start_date', y_col='weekly_approval_rating_percent',
                    group_col='politician_name', title='Weekly Approval Rating', xlabel='Week Start Date', ylabel='Approval Rating (%)'
                )
                weekly_trend_img_base64 = get_image_as_base64(weekly_trend_img_buf)
        
        df_display_ready = pd.DataFrame()
        if not weekly_df_multiple.empty:
            cols_to_display = ['politician_name', 'year_week', 'week_start_date', 'weekly_approval_rating_percent', 'total_votes_in_week']
            actual_cols = [col for col in cols_to_display if col in weekly_df_multiple.columns]
            if actual_cols: df_display_ready = weekly_df_multiple[actual_cols].copy()
        
        trends_data_dict = {
            'all_politicians': politicians_list_df,
            'selected_politician_ids': selected_politician_ids,
            'selected_politician_names': selected_politician_names, 
            'weekly_df': weekly_df_multiple,
            'weekly_trend_img_base64': weekly_trend_img_base64,
            'df_display_ready': df_display_ready
        }
    elif active_tab == 'similarity':
        MAX_HEATMAP_POLITICIANS_CONST = 30
        df_all_for_selection = fetch_sentiment_distribution_per_politician(
            engine, min_total_votes_threshold=1, sort_by_total_votes=True
        )
        available_politicians_for_similarity = []
        if not df_all_for_selection.empty:
            available_politicians_for_similarity = df_all_for_selection[['politician_id', 'politician_name', 'total_votes']].to_dict(orient='records')
        
        selected_politician_ids_str = request.args.getlist('politician_ids_similarity')
        selected_politician_ids = [int(pid) for pid in selected_politician_ids_str if pid.isdigit()]

        if not selected_politician_ids_str and not df_all_for_selection.empty:
             if not df_all_for_selection.empty:
                selected_politician_ids = df_all_for_selection['politician_id'].head(min(5, len(df_all_for_selection))).tolist()
        
        ids_for_heatmap_calc = []
        if "All" in request.args.get('politician_select_mode_similarity', '') and not df_all_for_selection.empty:
             ids_for_heatmap_calc = df_all_for_selection['politician_id'].tolist()
        else:
             ids_for_heatmap_calc = selected_politician_ids

        heatmap_img_base64 = None
        similarity_df_valence_html = None
        df_for_similarity_calc = pd.DataFrame()
        
        if ids_for_heatmap_calc and not df_all_for_selection.empty:
            df_selected_full = df_all_for_selection[df_all_for_selection['politician_id'].isin(ids_for_heatmap_calc)]
            df_for_similarity_calc = df_selected_full.head(MAX_HEATMAP_POLITICIANS_CONST).copy()
            
            if not df_for_similarity_calc.empty and len(df_for_similarity_calc) > 1:
                names = df_for_similarity_calc['politician_name'].tolist()
                vectors = df_for_similarity_calc[['approve_percent', 'neutral_percent', 'disapprove_percent']].values
                if vectors.ndim == 2 and vectors.shape[0] > 1:
                    sim_matrix = cosine_similarity(vectors)
                    sim_df = pd.DataFrame(sim_matrix, index=names, columns=names)
                    heatmap_buf = plot_similarity_heatmap_to_image(sim_df, title="Sentiment Similarity Matrix")
                    heatmap_img_base64 = get_image_as_base64(heatmap_buf)
                    similarity_df_valence_html = sim_df.style.format("{:.3f}").to_html(classes='styled-table', border=0)
        
        similarity_data_dict = {
            'available_politicians': available_politicians_for_similarity, 
            'selected_politician_ids': selected_politician_ids, 
            'df_for_similarity_calc': df_for_similarity_calc,
            'heatmap_img_base64': heatmap_img_base64,
            'similarity_df_html': similarity_df_valence_html,
            'MAX_HEATMAP_POLITICIANS': MAX_HEATMAP_POLITICIANS_CONST
        }
    elif active_tab == 'dataset':
        metrics = fetch_dataset_metrics(engine)
        dataset_data_dict = {
            'metrics': metrics
        }
    elif active_tab == 'feed': # Renamed tab
        feed_df = fetch_feed_updates(engine, limit=50) 
        
        feed_list_for_template = []
        if not feed_df.empty:
            feed_list_for_template = feed_df.to_dict(orient='records')
            for item in feed_list_for_template:
                if isinstance(item.get('Timestamp'), pd.Timestamp):
                    item['Timestamp'] = item['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(item.get('Timestamp'), datetime.datetime):
                    item['Timestamp'] = item['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        feed_data_dict = { # Renamed dictionary
            'latest_feed_items': feed_list_for_template 
        }


    return render_template('index.html',
                           active_tab=active_tab,
                           approval_data=approval_data_dict,
                           trends_data=trends_data_dict,
                           similarity_data=similarity_data_dict,
                           dataset_data=dataset_data_dict,
                           feed_data=feed_data_dict, # Pass renamed dict
                           engine_available=bool(engine)
                           )

# --- Favicon Route ---
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# --- Custom 404 Error Handler ---
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    if not debug_mode and DEPLOY_ENV == 'DEVELOPMENT':
        debug_mode = True
        app.logger.info("Forcing debug mode ON for local 'python3 app.py' execution as FLASK_ENV was not 'development'.")
    
    port_num = int(os.environ.get('PORT', 5000))
    app.logger.info(f"Attempting to run app with debug_mode: {debug_mode} on port {port_num}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port_num)