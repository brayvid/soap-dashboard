# Copyright 2024-2025 soap.fyi. All rights reserved. <https://soap.fyi> */

import matplotlib
matplotlib.use('Agg')

from dotenv import load_dotenv
load_dotenv() # Load .env before other imports that might use env vars

import os
from flask import Flask, render_template, request, send_from_directory, redirect
import pandas as pd
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import base64
import datetime
import spacy # Semantic comparison
from matplotlib.colors import LinearSegmentedColormap # Custom colormaps
import matplotlib.dates as mdates
from urllib.parse import urljoin, urlencode # For constructing canonical URLs

# --- SPEED OPTIMIZATIONS: Import required libraries ---
from flask_compress import Compress
from flask_assets import Environment, Bundle
# --------------------------------------------------------

app = Flask(__name__)

# --- Determine Deployment Environment ---
DEPLOY_ENV = os.environ.get('DEPLOY_ENV', 'DEVELOPMENT').upper()

# ===================================================================
# --- FIX 1: REDIRECT WWW TO NON-WWW ---
# This middleware runs before every request to enforce the canonical domain.
# It should always be active, not just in production.
# ===================================================================
@app.before_request
def redirect_to_non_www():
    """Redirect www requests to non-www to enforce canonical URL."""
    # Check if the request is on the www subdomain and not a health check
    if request.host.startswith('www.') and request.path != '/healthz':
        # Construct the new URL by removing 'www.'
        new_url = request.url.replace('www.', '', 1)
        # Issue a permanent redirect (301)
        app.logger.info(f"Redirecting from www to non-www: {request.url} -> {new_url}")
        return redirect(new_url, code=301)

# ===================================================================
# --- FIX 2: AUTOMATICALLY PROVIDE CORRECT CANONICAL URL TO ALL TEMPLATES ---
# This context processor makes `canonical_url` available in every template.
# It is now environment-aware.
# ===================================================================
@app.context_processor
def inject_canonical_url():
    """
    Makes the canonical URL for the current page available to all templates.
    Uses a configured production URL in DEPLOY_ENV='PRODUCTION', otherwise omits it.
    """
    if DEPLOY_ENV == 'PRODUCTION':
        canonical_base_url = os.environ.get('DASHBOARD_CANONICAL_BASE_URL')
        if canonical_base_url:
            # Reconstruct the URL using the canonical host and the current path
            canonical_url_full = urljoin(canonical_base_url, request.path)

            # Add query string if present
            if request.query_string:
                canonical_url_full += '?' + request.query_string.decode('utf-8')
            
            app.logger.debug(f"Canonical URL injected for PRODUCTION: {canonical_url_full}")
            return dict(canonical_url=canonical_url_full)
        else:
            app.logger.warning("DASHBOARD_CANONICAL_BASE_URL not set in PRODUCTION. Canonical tag will be omitted.")
            return dict(canonical_url=None) # Explicitly set to None
    else:
        # In development or other non-production environments, do not render a canonical tag.
        # This prevents incorrect local hostnames from being used.
        app.logger.debug(f"Canonical tag omitted for non-PRODUCTION environment (current host: {request.host}).")
        return dict(canonical_url=None)


# --- SPEED OPTIMIZATIONS START ---
Compress(app)
assets = Environment(app)
assets.url = app.static_url_path
css_bundle = Bundle('styles.css', filters='cssmin', output='gen/packed.css')
assets.register('css_all', css_bundle)
# --- SPEED OPTIMIZATIONS END ---

# --- SpaCy Model Loading ---
try:
    nlp = spacy.load('en_core_web_md')
    SPACY_MODEL_LOADED = True
    app.logger.info("Successfully loaded spaCy model 'en_core_web_md'.")
except OSError:
    app.logger.error("spaCy model 'en_core_web_md' not found.")
    nlp = None
    SPACY_MODEL_LOADED = False

# --- Database Connection ---
def get_env_var(var_name_prefix, key):
    # Base variable name, e.g., DB_USERNAME
    base_var_name = f"{var_name_prefix}_{key.upper()}"
    
    # Production-specific variable name, e.g., DB_USERNAME_PROD
    prod_var_name = f"{base_var_name}_PROD"

    if DEPLOY_ENV == 'PRODUCTION':
        # Try the _PROD variable first (this will likely be None in your case).
        val = os.environ.get(prod_var_name)
        
        # If the _PROD variable is not set, fall back to the base variable.
        # This is the critical change: it now uses DB_USERNAME, DB_PASSWORD, etc.
        if val is None:
            val = os.environ.get(base_var_name)
            if val is None:
                 # Log a warning if neither is found, indicating a configuration issue.
                 app.logger.error(f"CRITICAL: Missing database configuration for: '{prod_var_name}' and '{base_var_name}' (DEPLOY_ENV: {DEPLOY_ENV})")
            # else: # Optionally log if fallback was successful
            #      app.logger.debug(f"Using fallback: '{base_var_name}' for DB configuration in PRODUCTION.")
        return val
    else:
        # For non-production, just use the base variable.
        val = os.environ.get(base_var_name)
        if val is None:
            # Log a warning if the base variable is missing in non-prod too.
            app.logger.warning(f"Missing database configuration for: {base_var_name} (DEPLOY_ENV: {DEPLOY_ENV})")
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
        if DEPLOY_ENV == 'PRODUCTION':
             db_connection_str += '?sslmode=require'
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

def fetch_sentiment_distribution_per_politician(_engine, min_total_votes_threshold=20, sort_by_total_votes=False):
    if not _engine: return pd.DataFrame()
    if sort_by_total_votes:
        order_by_clause_final = "ORDER BY total_votes DESC, calculated_ranking_score DESC, politician_name ASC"
    else:
        order_by_clause_final = "ORDER BY calculated_ranking_score DESC, total_votes DESC, politician_name ASC"
    query_with_final_order_by = text(f"""
        WITH VoteSentimentScores AS (
            SELECT v.politician_id, w.sentiment_score
            FROM votes AS v JOIN words AS w ON v.word_id = w.word_id
            WHERE w.sentiment_score IS NOT NULL
        ), PoliticianAggregates AS (
            SELECT
                p.politician_id,
                p.name AS politician_name,
                COUNT(vss.sentiment_score) AS total_votes,
                COALESCE(SUM(vss.sentiment_score) / NULLIF(COUNT(vss.sentiment_score), 0), 0) AS average_sentiment_score,
                (COALESCE(SUM(vss.sentiment_score) / NULLIF(COUNT(vss.sentiment_score), 0), 0) * 100) AS calculated_ranking_score
            FROM politicians AS p
            LEFT JOIN VoteSentimentScores AS vss ON p.politician_id = vss.politician_id
            GROUP BY p.politician_id, p.name
            HAVING COUNT(vss.sentiment_score) >= :min_votes_threshold
        )
        SELECT
            politician_id,
            politician_name,
            total_votes,
            calculated_ranking_score
        FROM PoliticianAggregates
        {order_by_clause_final};
    """)
    try:
        with _engine.connect() as connection:
            df = pd.read_sql(query_with_final_order_by, connection, params={'min_votes_threshold': min_total_votes_threshold})
        for col in ['total_votes', 'calculated_ranking_score']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except Exception as e:
        app.logger.error(f"Error in fetch_sentiment_distribution_per_politician: {e}\nQuery was: {query_with_final_order_by}")
        return pd.DataFrame()

def fetch_raw_sentiments_for_multiple_politicians(_engine, politician_ids_list):
    df_cols = ["politician_id", "politician_name", "sentiment_score"]
    if not _engine or not politician_ids_list: return pd.DataFrame(columns=df_cols)
    try: safe_politician_ids = tuple(map(int, politician_ids_list))
    except (ValueError, TypeError):
        app.logger.error(f"Invalid politician IDs for raw sentiment fetch: {politician_ids_list}")
        return pd.DataFrame(columns=df_cols)
    if not safe_politician_ids: return pd.DataFrame(columns=df_cols)
    query = text("""
        SELECT p.politician_id, p.name as politician_name, w.sentiment_score
        FROM votes v
        JOIN words w ON v.word_id = w.word_id
        JOIN politicians p ON v.politician_id = p.politician_id
        WHERE v.politician_id IN :politician_ids_param
          AND w.sentiment_score IS NOT NULL;
    """)
    try:
        with _engine.connect() as connection:
            df = pd.read_sql(query, connection, params={'politician_ids_param': safe_politician_ids})
        if 'sentiment_score' in df.columns:
            df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
            df.dropna(subset=['sentiment_score'], inplace=True)
        return df
    except Exception as e:
        app.logger.error(f"Error fetching raw sentiments for multiple politicians: {e}")
        return pd.DataFrame(columns=df_cols)

def fetch_word_counts_per_politician(_engine, politician_ids_list=None):
    df_cols = ["politician_id", "politician_name", "word", "count"]
    if not _engine: return pd.DataFrame(columns=df_cols)
    where_clause = ""
    params = {}
    if politician_ids_list:
        try:
            safe_politician_ids = tuple(map(int, politician_ids_list))
            if safe_politician_ids:
                where_clause = "WHERE p.politician_id IN :p_ids"
                params['p_ids'] = safe_politician_ids
        except (ValueError, TypeError):
            app.logger.error(f"Invalid politician IDs provided for word count fetch: {politician_ids_list}")
            return pd.DataFrame(columns=df_cols)
    query = text(f"""
        SELECT
            p.politician_id, p.name AS politician_name, w.word, COUNT(v.vote_id) AS "count"
        FROM votes v
        JOIN politicians p ON v.politician_id = p.politician_id
        JOIN words w ON v.word_id = w.word_id
        {where_clause}
        GROUP BY p.politician_id, p.name, w.word
        ORDER BY p.politician_id, "count" DESC;
    """)
    try:
        with _engine.connect() as connection: df = pd.read_sql(query, connection, params=params)
        return df
    except Exception as e:
        app.logger.error(f"Error fetching word counts: {e}")
        return pd.DataFrame(columns=df_cols)

def fetch_weekly_approval_rating(_engine, politician_ids_list):
    if not _engine or not politician_ids_list: return pd.DataFrame()
    try: safe_politician_ids = tuple(map(int, politician_ids_list))
    except ValueError:
        app.logger.error(f"Invalid non-integer ID in politician_ids_list for weekly approval: {politician_ids_list}")
        return pd.DataFrame()
    if not safe_politician_ids: return pd.DataFrame()
    in_clause_sql = f"({safe_politician_ids[0]})" if len(safe_politician_ids) == 1 else str(safe_politician_ids)
    query = text(f"""
        SELECT p.name AS politician_name, p.politician_id,
            TO_CHAR(v.created_at, 'IYYY-IW') AS year_week,
            DATE_TRUNC('week', v.created_at)::date AS week_start_date,
            COUNT(v.vote_id) AS total_votes_in_week,
            CASE WHEN COUNT(v.vote_id) > 0 THEN (((SUM(w.sentiment_score) / COUNT(v.vote_id)) / 2.0) + 0.5) * 100.0 ELSE NULL END AS weekly_approval_rating_percent
        FROM votes AS v JOIN words AS w ON v.word_id = w.word_id JOIN politicians AS p ON v.politician_id = p.politician_id
        WHERE v.politician_id IN {in_clause_sql} AND w.sentiment_score IS NOT NULL AND v.created_at IS NOT NULL
        GROUP BY p.politician_id, p.name, year_week, week_start_date ORDER BY p.name ASC, week_start_date ASC;""")
    try:
        with _engine.connect() as connection: df = pd.read_sql(query, connection)
        if df.empty: return pd.DataFrame(columns=['politician_name', 'politician_id', 'year_week', 'week_start_date', 'total_votes_in_week', 'weekly_approval_rating_percent'])
        if 'weekly_approval_rating_percent' in df.columns: df['weekly_approval_rating_percent'] = pd.to_numeric(df['weekly_approval_rating_percent'], errors='coerce')
        if 'week_start_date' in df.columns: df['week_start_date'] = pd.to_datetime(df['week_start_date'], errors='coerce')
        if 'total_votes_in_week' in df.columns: df['total_votes_in_week'] = pd.to_numeric(df['total_votes_in_week'], errors='coerce').fillna(0).astype(int)
        expected_cols = ['politician_name', 'politician_id', 'year_week', 'week_start_date', 'total_votes_in_week', 'weekly_approval_rating_percent']
        for col in expected_cols:
            if col not in df.columns: df[col] = pd.NA
        return df
    except Exception as e:
        app.logger.error(f"Error in fetch_weekly_approval_rating: {e}\nQuery: {query}")
        return pd.DataFrame()

def fetch_daily_activity(_engine):
    """Fetches daily counts of word submissions, new politician additions, and new word additions."""
    if not _engine: return pd.DataFrame()

    submissions_query = text("""
        SELECT DATE(created_at) AS activity_date, COUNT(vote_id) AS submission_count
        FROM votes WHERE created_at IS NOT NULL GROUP BY activity_date;
    """)
    additions_query = text("""
        SELECT DATE(created_at) AS activity_date, COUNT(politician_id) AS addition_count
        FROM politicians WHERE created_at IS NOT NULL GROUP BY activity_date;
    """)
    new_words_query = text("""
        SELECT DATE(created_at) AS activity_date, COUNT(word_id) AS new_word_count
        FROM words WHERE created_at IS NOT NULL GROUP BY activity_date;
    """)

    try:
        with _engine.connect() as connection:
            submissions_df = pd.read_sql(submissions_query, connection)
            additions_df = pd.read_sql(additions_query, connection)
            new_words_df = pd.read_sql(new_words_query, connection)

        for df in [submissions_df, additions_df, new_words_df]:
            if not df.empty:
                df['activity_date'] = pd.to_datetime(df['activity_date'])

        if submissions_df.empty and additions_df.empty and new_words_df.empty:
            return pd.DataFrame(columns=['activity_date', 'submission_count', 'addition_count', 'new_word_count'])

        # Merge all three dataframes
        activity_df = pd.merge(submissions_df, additions_df, on='activity_date', how='outer')
        activity_df = pd.merge(activity_df, new_words_df, on='activity_date', how='outer')

        activity_df.fillna(0, inplace=True)
        activity_df['submission_count'] = activity_df['submission_count'].astype(int)
        activity_df['addition_count'] = activity_df['addition_count'].astype(int)
        activity_df['new_word_count'] = activity_df['new_word_count'].astype(int)
        activity_df.sort_values('activity_date', inplace=True)
        return activity_df

    except Exception as e:
        app.logger.error(f"Error in fetch_daily_activity: {e}")
        return pd.DataFrame()
    
def fetch_dataset_metrics(_engine):
    metric_keys = [
        "total_politicians", "total_words_scorable", "total_votes", "votes_date_range",
        "net_sentiment_sum_all_submissions", "average_sentiment_score_all_submissions",
        "net_approval_rating_percent_all_submissions", "avg_submissions_per_politician",
        "submissions_last_7_days", "most_active_day", "most_described_politician",
        "highest_rated_politician", "lowest_rated_politician",
        "most_submitted_word", "most_submitted_word_last_7_days",
        "most_positive_word_attribution", "most_negative_word_attribution"
    ]
    if not _engine: return {key: "N/A" for key in metric_keys}
    metrics = {key: "N/A" for key in metric_keys}
    MIN_VOTES_FOR_RATING = 20
    try:
        with _engine.connect() as connection:
            total_politicians_num = connection.execute(text("SELECT COUNT(*) FROM politicians;")).scalar_one_or_none() or 0
            total_votes_num = connection.execute(text("SELECT COUNT(*) FROM votes;")).scalar_one_or_none() or 0
            total_words_scorable_num = connection.execute(text("SELECT COUNT(*) FROM words WHERE sentiment_score IS NOT NULL;")).scalar_one_or_none() or 0
            
            metrics["total_politicians"] = f"{total_politicians_num:,}"
            metrics["total_votes"] = f"{total_votes_num:,}"
            metrics["total_words_scorable"] = f"{total_words_scorable_num:,}"

            res_dates = connection.execute(text("SELECT MIN(created_at)::date, MAX(created_at)::date FROM votes;")).fetchone()
            if res_dates and res_dates[0] and res_dates[1]:
                metrics["votes_date_range"] = f"{res_dates[0].strftime('%Y-%m-%d')} to {res_dates[1].strftime('%Y-%m-%d')}"
            
            if total_votes_num > 0 and total_politicians_num > 0:
                metrics["avg_submissions_per_politician"] = f"{(total_votes_num / total_politicians_num):.1f}"

            try:
                metrics["submissions_last_7_days"] = f"{connection.execute(text('''SELECT COUNT(*) FROM votes WHERE created_at >= NOW() - INTERVAL '7 days';''')).scalar_one():,}"
            except Exception as e: app.logger.error(f"Error fetching submissions_last_7_days: {e}")
            
            sentiment_query = text("SELECT SUM(w.sentiment_score), COUNT(v.vote_id) FROM votes v JOIN words w ON v.word_id = w.word_id WHERE w.sentiment_score IS NOT NULL;")
            sentiment_res = connection.execute(sentiment_query).fetchone()
            if sentiment_res and sentiment_res[1] and sentiment_res[1] > 0:
                total_sum, count_votes = sentiment_res[0], sentiment_res[1]
                metrics["net_sentiment_sum_all_submissions"] = f"{total_sum:.2f}"
                avg_score = total_sum / count_votes
                metrics["average_sentiment_score_all_submissions"] = f"{avg_score:.4f}"
                approval_percent = (((avg_score / 2.0) + 0.5) * 100.0)
                metrics["net_approval_rating_percent_all_submissions"] = f"{approval_percent:.2f}%"
            try:
                res = connection.execute(text("""
                    WITH r AS (SELECT DATE(created_at) d, COUNT(*) c, RANK() OVER (ORDER BY COUNT(*) DESC) rnk FROM votes GROUP BY d)
                    SELECT d, c FROM r WHERE rnk = 1;
                """)).fetchall()
                if res: metrics["most_active_day"] = ", ".join([f"{d.strftime('%Y-%m-%d')} ({c} submissions)" for d, c in res])
            except Exception as e: app.logger.error(f"Error fetching most_active_day: {e}")
            try:
                res = connection.execute(text("""
                    WITH r AS (SELECT p.politician_id i, p.name n, COUNT(v.vote_id) c, RANK() OVER (ORDER BY COUNT(v.vote_id) DESC) rnk FROM votes v JOIN politicians p ON v.politician_id = p.politician_id GROUP BY p.politician_id, p.name)
                    SELECT i, n, c FROM r WHERE rnk = 1;
                """)).fetchall()
                if res: metrics["most_described_politician"] = ", ".join([f'<a href="https://use.soap.fyi/politician/{i}">{n}</a> ({c:,} submissions)' for i, n, c in res])
            except Exception as e: app.logger.error(f"Error fetching most_described_politician: {e}")
            try:
                res = connection.execute(text("""
                    WITH s AS (SELECT p.politician_id i, p.name n, (SUM(w.sentiment_score)/COUNT(v.vote_id)) sc, RANK() OVER (ORDER BY (SUM(w.sentiment_score)/COUNT(v.vote_id)) DESC) r_hi, RANK() OVER (ORDER BY (SUM(w.sentiment_score)/COUNT(v.vote_id)) ASC) r_lo FROM votes v JOIN words w ON v.word_id=w.word_id JOIN politicians p ON v.politician_id=p.politician_id WHERE w.sentiment_score IS NOT NULL GROUP BY p.politician_id, p.name HAVING COUNT(v.vote_id) >= :min_votes)
                    SELECT i, n, sc, r_hi, r_lo FROM s WHERE r_hi=1 OR r_lo=1;
                """), {'min_votes': MIN_VOTES_FOR_RATING}).fetchall()
                if res:
                    hi_rated = [f'<a href="https://use.soap.fyi/politician/{r.i}">{r.n}</a> ({(((r.sc/2.0)+0.5)*100.0):.1f}%)' for r in res if r.r_hi==1]
                    lo_rated = [f'<a href="https://use.soap.fyi/politician/{r.i}">{r.n}</a> ({(((r.sc/2.0)+0.5)*100.0):.1f}%)' for r in res if r.r_lo==1]
                    if hi_rated: metrics["highest_rated_politician"] = ", ".join(hi_rated)
                    if lo_rated: metrics["lowest_rated_politician"] = ", ".join(lo_rated)
            except Exception as e: app.logger.error(f"Error fetching politician_ratings: {e}")

            # --- START: Most Submitted Word (All Time) ---
            try:
                query_all_time = text("""
                    WITH WordTotalRank AS (
                        SELECT w.word, COUNT(v.vote_id) as total_count, RANK() OVER (ORDER BY COUNT(v.vote_id) DESC) as rnk
                        FROM votes v JOIN words w ON v.word_id = w.word_id GROUP BY w.word
                    ), TopWords AS (
                        SELECT word FROM WordTotalRank WHERE rnk = 1
                    ), PoliticianWordCounts AS (
                        SELECT w.word, p.politician_id, p.name as politician_name, COUNT(v.vote_id) as politician_word_count,
                               ROW_NUMBER() OVER (PARTITION BY w.word ORDER BY COUNT(v.vote_id) DESC, p.name ASC) as rn
                        FROM votes v
                        JOIN words w ON v.word_id = w.word_id JOIN politicians p ON v.politician_id = p.politician_id
                        WHERE w.word IN (SELECT word FROM TopWords)
                        GROUP BY w.word, p.politician_id, p.name
                    )
                    SELECT word, STRING_AGG('<a href="https://use.soap.fyi/politician/' || politician_id || '">' || politician_name || '</a> (' || politician_word_count || ')', ', ' ORDER BY politician_word_count DESC, politician_name ASC) as politicians_breakdown
                    FROM PoliticianWordCounts 
                    WHERE rn <= 3
                    GROUP BY word;
                """)
                res = connection.execute(query_all_time).fetchall()
                if res:
                    # Fetch all words with their sentiment scores once
                    word_sentiments_query = text("SELECT word, sentiment_score FROM words WHERE sentiment_score IS NOT NULL")
                    word_sentiments_list = connection.execute(word_sentiments_query).fetchall()
                    word_sentiments = {r.word: r.sentiment_score for r in word_sentiments_list}

                    formatted_results = [f"{row.word.capitalize()} ({word_sentiments.get(row.word, 0):+.2f}): {row.politicians_breakdown}" for row in res]
                    metrics["most_submitted_word"] = "; ".join(formatted_results)
            except Exception as e: app.logger.error(f"Error fetching most_submitted_word: {e}")
            # --- END: Most Submitted Word (All Time) ---

            # --- START: Most Submitted Word (Last 7 Days) ---
            try:
                query_last_7_days = text("""
                    WITH WordTotalRank AS (
                        SELECT w.word, COUNT(v.vote_id) as total_count, RANK() OVER (ORDER BY COUNT(v.vote_id) DESC) as rnk
                        FROM votes v JOIN words w ON v.word_id = w.word_id
                        WHERE v.created_at >= NOW() - INTERVAL '7 days' GROUP BY w.word
                    ), TopWords AS (
                        SELECT word FROM WordTotalRank WHERE rnk = 1
                    ), PoliticianWordCounts AS (
                        SELECT w.word, p.politician_id, p.name as politician_name, COUNT(v.vote_id) as politician_word_count,
                               ROW_NUMBER() OVER (PARTITION BY w.word ORDER BY COUNT(v.vote_id) DESC, p.name ASC) as rn
                        FROM votes v
                        JOIN words w ON v.word_id = w.word_id JOIN politicians p ON v.politician_id = p.politician_id
                        WHERE w.word IN (SELECT word FROM TopWords) AND v.created_at >= NOW() - INTERVAL '7 days'
                        GROUP BY w.word, p.politician_id, p.name
                    )
                    SELECT word, STRING_AGG('<a href="https://use.soap.fyi/politician/' || politician_id || '">' || politician_name || '</a> (' || politician_word_count || ')', ', ' ORDER BY politician_word_count DESC, politician_name ASC) as politicians_breakdown
                    FROM PoliticianWordCounts 
                    WHERE rn <= 3
                    GROUP BY word;
                """)
                res = connection.execute(query_last_7_days).fetchall()
                if res:
                    # Fetch all words with their sentiment scores once
                    word_sentiments_query = text("SELECT word, sentiment_score FROM words WHERE sentiment_score IS NOT NULL")
                    word_sentiments_list = connection.execute(word_sentiments_query).fetchall()
                    word_sentiments = {r.word: r.sentiment_score for r in word_sentiments_list}

                    formatted_results = [f"{row.word.capitalize()} ({word_sentiments.get(row.word, 0):+.2f}): {row.politicians_breakdown}" for row in res]
                    metrics["most_submitted_word_last_7_days"] = "; ".join(formatted_results)
                else:
                    metrics["most_submitted_word_last_7_days"] = "No submissions in last 7 days"
            except Exception as e: app.logger.error(f"Error fetching most_submitted_word_last_7_days: {e}")
            # --- END: Most Submitted Word (Last 7 Days) ---

            # --- START: Most Positive/Negative Word Attribution (with counts) ---
            try:
                extreme_words_query = text("""
                    SELECT word, sentiment_score FROM words WHERE sentiment_score IS NOT NULL AND (
                        sentiment_score = (SELECT MAX(sentiment_score) FROM words WHERE sentiment_score IS NOT NULL)
                        OR
                        sentiment_score = (SELECT MIN(sentiment_score) FROM words WHERE sentiment_score IS NOT NULL)
                    );
                """)
                extreme_words_res = connection.execute(extreme_words_query).fetchall()
                if extreme_words_res:
                    all_scores = [w.sentiment_score for w in extreme_words_res]
                    max_score, min_score = max(all_scores), min(all_scores)
                    positive_words = {w.word for w in extreme_words_res if w.sentiment_score == max_score}
                    negative_words = {w.word for w in extreme_words_res if w.sentiment_score == min_score}
                    all_extreme_words_list = list(positive_words | negative_words)
                    
                    if all_extreme_words_list:
                        attribution_query = text("""
                            SELECT w.word, p.politician_id, p.name as politician_name, COUNT(v.vote_id) as word_count
                            FROM votes v
                            JOIN politicians p ON v.politician_id = p.politician_id
                            JOIN words w ON v.word_id = w.word_id
                            WHERE w.word IN :word_list
                            GROUP BY w.word, p.politician_id, p.name
                            ORDER BY w.word, word_count DESC, p.name
                        """)
                        attribution_df = pd.read_sql(attribution_query, connection, params={'word_list': tuple(all_extreme_words_list)})

                        pos_attributions = []
                        for word in sorted(list(positive_words)):
                            politicians_df_for_word = attribution_df[attribution_df['word'] == word]
                            if not politicians_df_for_word.empty:
                                details = [f'<a href="https://use.soap.fyi/politician/{row.politician_id}">{row.politician_name}</a> ({row.word_count})' for row in politicians_df_for_word.head(3).itertuples()]
                                pos_attributions.append(f"{word.capitalize()} ({max_score:+.2f}): {', '.join(details)}")
                        if pos_attributions:
                            metrics["most_positive_word_attribution"] = "; ".join(pos_attributions)

                        neg_attributions = []
                        for word in sorted(list(negative_words)):
                            politicians_df_for_word = attribution_df[attribution_df['word'] == word]
                            if not politicians_df_for_word.empty:
                                details = [f'<a href="https://use.soap.fyi/politician/{row.politician_id}">{row.politician_name}</a> ({row.word_count})' for row in politicians_df_for_word.head(3).itertuples()]
                                neg_attributions.append(f"{word.capitalize()} ({min_score:+.2f}): {', '.join(details)}")
                        if neg_attributions:
                            metrics["most_negative_word_attribution"] = "; ".join(neg_attributions)

            except Exception as e:
                app.logger.error(f"Error fetching extreme word attribution: {e}")
            # --- END: Most Positive/Negative Word Attribution (with counts) ---

    except Exception as e:
        app.logger.error(f"Major error in fetch_dataset_metrics: {e}")
        return {key: "Error" for key in metric_keys}
    return metrics

def fetch_feed_updates(_engine, limit=100):
    df_cols = ["Timestamp", "Politician", "Word", "Sentiment"]
    if not _engine: return pd.DataFrame(columns=df_cols)
    query = text(f"""
        SELECT
            v.created_at AS "Timestamp", p.name AS "Politician", w.word AS "Word", w.sentiment_score AS "Sentiment"
        FROM votes v JOIN words w ON v.word_id = w.word_id JOIN politicians p ON v.politician_id = p.politician_id
        ORDER BY v.created_at DESC
        LIMIT :limit;
    """)
    try:
        with _engine.connect() as connection:
            df = pd.read_sql(query, connection, params={'limit': limit})
        if not df.empty:
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d')
            if 'Word' in df.columns:
                df['Word'] = df['Word'].astype(str).apply(lambda x: ' '.join(s.capitalize() for s in x.split()))
            if 'Sentiment' in df.columns:
                df['Sentiment'] = pd.to_numeric(df['Sentiment'], errors='coerce')
            
            for col in df_cols:
                if col not in df.columns: df[col] = pd.NA
            df = df[df_cols]
        else:
            df = pd.DataFrame(columns=df_cols)
        return df
    except Exception as e:
        app.logger.error(f"Error fetching feed updates: {e}\nQuery: {query}")
        return pd.DataFrame(columns=df_cols)

def fetch_top_weekly_word_for_politician(_engine, politician_id):
    # Define the new column names as requested.
    df_cols = ["Week Start", "Top Word", "Top Word Votes", "All Votes"]
    if not _engine or politician_id is None: return pd.DataFrame(columns=df_cols)
    try: pid = int(politician_id)
    except ValueError:
        app.logger.error(f"Invalid politician_id for top weekly word: {politician_id}")
        return pd.DataFrame(columns=df_cols)
    
    # Updated SQL query to remove the 'year_week' and rename the output columns.
    query = text(f"""
        WITH WeeklyWordCounts AS (
            SELECT v.politician_id, w.word AS word_text, DATE_TRUNC('week', v.created_at)::date AS week_start_date,
                   COUNT(v.vote_id) AS word_vote_count
            FROM votes v JOIN words w ON v.word_id = w.word_id WHERE v.politician_id = :politician_id_param
            GROUP BY v.politician_id, w.word, week_start_date
        ), RankedWeeklyWords AS (
            SELECT politician_id, word_text, week_start_date, word_vote_count,
                   ROW_NUMBER() OVER (PARTITION BY politician_id, week_start_date ORDER BY word_vote_count DESC, word_text ASC) as rn
            FROM WeeklyWordCounts
        ), TotalVotesPerWeek AS (
            SELECT politician_id, DATE_TRUNC('week', v.created_at)::date AS week_start_date, COUNT(v.vote_id) as total_weekly_votes
            FROM votes v WHERE v.politician_id = :politician_id_param GROUP BY politician_id, week_start_date
        )
        SELECT rww.week_start_date AS "Week Start", rww.word_text AS "Top Word",
               rww.word_vote_count AS "Top Word Votes", COALESCE(tvpw.total_weekly_votes, 0) AS "All Votes"
        FROM RankedWeeklyWords rww LEFT JOIN TotalVotesPerWeek tvpw
            ON rww.politician_id = tvpw.politician_id AND rww.week_start_date = tvpw.week_start_date
        WHERE rww.rn = 1 ORDER BY rww.week_start_date DESC;""")
    try:
        with _engine.connect() as connection: df = pd.read_sql(query, connection, params={'politician_id_param': pid})
        if df.empty: return pd.DataFrame(columns=df_cols)

        # Update post-processing to use the new column names.
        if "Week Start" in df.columns: df["Week Start"] = pd.to_datetime(df["Week Start"])
        if "Top Word" in df.columns: df["Top Word"] = df["Top Word"].astype(str).apply(lambda x: ' '.join(s.capitalize() for s in x.split()))
        for col in ["Top Word Votes", "All Votes"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Ensure all expected columns are present before returning
        for col in df_cols:
            if col not in df.columns:
                df[col] = pd.NA
        return df[df_cols]

    except Exception as e:
        app.logger.error(f"Error fetching top weekly word for politician_id {pid}: {e}\nQuery: {query}")
        return pd.DataFrame(columns=df_cols)

# --- Plotting Functions ---
SENTIMENT_COLORMAP = LinearSegmentedColormap.from_list("sentiment_spectrum", [
    (0.0,    '#DE3B3B'), (0.475,  '#CDb14c'), (0.475,  '#BFBFBF'), (0.525,  '#BFBFBF'),
    (0.525,  '#9fad42'), (1.0,    '#2a8d64')
])
def plot_multi_sentiment_histograms_to_image(df_scores_multi, num_bins=12):
    if df_scores_multi.empty or 'sentiment_score' not in df_scores_multi.columns: return None
    politicians_in_order = df_scores_multi['politician_name'].unique()
    num_politicians = len(politicians_in_order)
    if num_politicians == 0: return None
    bins = np.linspace(-1, 1, num_bins + 1)
    subplot_height = 2.8 if num_politicians <= 3 else 1.6
    total_fig_height = max(5, num_politicians * subplot_height)
    fig, axes = plt.subplots(nrows=num_politicians, ncols=1, figsize=(12, total_fig_height))
    if num_politicians == 1: axes = [axes]
    fig.suptitle('Sentiment Score Distribution', fontsize=18, weight='bold', y=0.995)
    for i, name in enumerate(politicians_in_order):
        ax = axes[i]
        politician_scores = df_scores_multi[df_scores_multi['politician_name'] == name]['sentiment_score']
        n, bins_edges, patches = ax.hist(politician_scores, bins=bins, edgecolor='black', zorder=2)
        for patch in patches:
            bin_center = (patch.get_x() + patch.get_x() + patch.get_width()) / 2
            normalized_score = np.clip((bin_center + 1) / 2, 0, 1)
            color = SENTIMENT_COLORMAP(normalized_score)
            patch.set_facecolor(color)
        if not politician_scores.empty:
            median_score = np.median(politician_scores)
            ax.axvline(median_score, color='red', linestyle='--', linewidth=1.5, label=f'Median: {median_score:.2f}', zorder=3)
        ax.legend(loc='upper right', frameon=False, fontsize=9)
        ax.set_title(f"{name}", fontsize=15, pad=5)
        ax.set_ylabel('Count', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6, zorder=1)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_xlim(-1, 1)
        if i == 0 or i == num_politicians - 1: ax.set_xlabel('Sentiment Score (-1 to +1)', fontsize=13)
        else:
            ax.set_xlabel(''); ax.tick_params(axis='x', labelbottom=True)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    img_buf = BytesIO(); fig.savefig(img_buf, format="png", dpi=100, bbox_inches='tight', pad_inches=0.1)
    img_buf.seek(0); plt.close(fig); return img_buf

def plot_multiline_chart_to_image(df, x_col, y_col, group_col, title, xlabel, ylabel, color_palette="tab20", decimal_places=0):
    if df.empty or x_col not in df.columns or y_col not in df.columns or group_col not in df.columns or df[y_col].isnull().all(): return None
    unique_groups = df[group_col].nunique(); fig_height = max(6, min(12, 5 + unique_groups * 0.3)); fig_width = 14
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    colors = sns.color_palette("tab10" if unique_groups <= 10 else color_palette, n_colors=unique_groups)
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

def plot_daily_activity_to_image(df):
    """Generates a dual-axis grouped bar chart for daily submissions, politician additions, and new word additions for the last 30 days."""
    import matplotlib.dates as mdates
    if df.empty or ('submission_count' not in df.columns and 'addition_count' not in df.columns):
        return None

    # Filter for the last 30 days directly.
    df_filtered = df[df['activity_date'] >= (datetime.datetime.now() - datetime.timedelta(days=30))]
    if df_filtered.empty:
        return None # Return if there is no data in the last 30 days.

    fig, ax1 = plt.subplots(figsize=(15, 7))
    bar_width = 0.25
    dates = df_filtered['activity_date']
    x_pos = mdates.date2num(dates)

    color1 = '#A997DF'  # Light Purple
    color2 = '#DC6E79'  # (220, 110, 121)
    color3 = '#46853E'  # (70, 133, 62)

    ax1.set_ylabel('Daily Votes', color=color1, fontsize=12, weight='bold')
    ax1.bar(x_pos - bar_width, df_filtered['submission_count'], width=bar_width, color=color1, label='Votes Submitted')
    ax1.tick_params(axis='y', labelcolor=color1)
    max_submissions = df_filtered['submission_count'].max()
    ax1.set_ylim(0, max(10, max_submissions * 1.1))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.spines['top'].set_visible(False)

    ax2 = ax1.twinx()
    
    ax2.set_ylabel('Daily Additions', fontsize=12, weight='bold')

    ax2.bar(x_pos, df_filtered['addition_count'], width=bar_width, color=color2, label='New Politicians Added')
    ax2.bar(x_pos + bar_width, df_filtered['new_word_count'], width=bar_width, color=color3, label='New Words Added')
    
    ax2.tick_params(axis='y')
    max_additions = df_filtered['addition_count'].max()
    max_new_words = df_filtered['new_word_count'].max()
    ax2.set_ylim(0, max(5, max(max_additions, max_new_words) * 1.2))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.spines['top'].set_visible(False)

    ax1.set_title('Daily Activity: Submissions & Additions (Last 30 Days)', fontsize=16, weight='bold', pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(rotation=30, ha='right')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    img_buf = BytesIO()
    fig.savefig(img_buf, format="png", dpi=100, bbox_inches='tight')
    img_buf.seek(0)
    plt.close(fig)
    return img_buf

def plot_comparison_heatmap_to_image(comparison_matrix_df, title="Comparison Matrix", cbar_label="Cosine Similarity"):
    if comparison_matrix_df.empty or len(comparison_matrix_df) < 2: return None
    plot_df = comparison_matrix_df.copy()
    display_n = len(plot_df)
    for col in plot_df.columns: plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    fig_height = max(8, min(25, 3 + display_n * 0.9)); fig_width = fig_height * 1.2
    if display_n < 5 : fig_height = 6; fig_width = 8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(plot_df, annot=True, fmt=".2f", cmap="viridis", linewidths=.5, ax=ax, cbar=True, square=True, annot_kws={"size": 6 if display_n > 20 else (7 if display_n > 15 else 8)}, cbar_kws={'label': cbar_label, 'shrink': 0.7})
    ax.set_title(title, fontsize=16, pad=20, weight='bold')
    ax.tick_params(axis='x', rotation=70, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=8)
    fig.subplots_adjust(left=0.3 if display_n > 5 else 0.25, bottom=0.3 if display_n > 5 else 0.25, right=0.98, top=0.95)
    img_buf = BytesIO(); fig.savefig(img_buf, format="png", bbox_inches='tight', dpi=120); img_buf.seek(0)
    plt.close(fig); return img_buf

def get_image_as_base64(img_buf):
    if img_buf:
        img_buf.seek(0)
        return base64.b64encode(img_buf.read()).decode('utf-8')
    return None

# --- Main Dashboard Route ---
@app.route('/')
def dashboard():
    if not engine:
        # Pass engine_available for the template to check (e.g., in base.html)
        return render_template('index.html', engine_available=False, active_tab='overview') 

    active_tab = request.args.get('tab', 'overview')

    politicians_list_df = fetch_politicians_list(engine)

    lookup_tab_data = {}
    compare_data_dict = {}
    overview_tab_data = {}
    feed_data_dict = {}

    if active_tab == 'lookup':
        politicians_by_votes_df = fetch_sentiment_distribution_per_politician(engine, min_total_votes_threshold=1, sort_by_total_votes=True)
        selected_politician_ids_str = request.args.getlist('politician_ids_lookup')
        select_all_mode = "All" in request.args.get('politician_select_mode_lookup', '')
        MAX_LOOKUP_POLITICIANS_CONST = 30
        selected_politician_ids = []
        if select_all_mode:
            if not politicians_by_votes_df.empty:
                selected_politician_ids = politicians_by_votes_df['politician_id'].head(MAX_LOOKUP_POLITICIANS_CONST).tolist()
        elif selected_politician_ids_str:
            selected_politician_ids = [int(pid) for pid in selected_politician_ids_str if pid.isdigit()]
        else:
            if not politicians_by_votes_df.empty:
                num_to_select = min(5, len(politicians_by_votes_df))
                selected_politician_ids = politicians_by_votes_df['politician_id'].head(num_to_select).tolist()

        ids_for_query = selected_politician_ids
        selected_politician_names = []
        if ids_for_query and not politicians_list_df.empty:
            selected_politician_names = politicians_list_df[politicians_list_df['politician_id'].isin(ids_for_query)]['name'].tolist()

        lookup_tab_data = {
            'all_politicians': politicians_list_df,
            'selected_politician_ids': selected_politician_ids,
            'selected_politician_names': selected_politician_names,
            'weekly_approval_img_base64': None,
            'histogram_img_base64': None,
            'top_word_tables': [],
            'MAX_LOOKUP_POLITICIANS': MAX_LOOKUP_POLITICIANS_CONST
        }
        if ids_for_query:
            weekly_df_multiple = fetch_weekly_approval_rating(engine, ids_for_query)
            if not weekly_df_multiple.empty and 'weekly_approval_rating_percent' in weekly_df_multiple.columns and weekly_df_multiple['weekly_approval_rating_percent'].notna().any():
                weekly_approval_img_buf = plot_multiline_chart_to_image(
                    weekly_df_multiple, x_col='week_start_date', y_col='weekly_approval_rating_percent',
                    group_col='politician_name', title='Weekly Approval Rating Trend', xlabel='Week Start Date', ylabel='Approval Rating (0-100%)')
                lookup_tab_data['weekly_approval_img_base64'] = get_image_as_base64(weekly_approval_img_buf)
            
            df_scores_raw = fetch_raw_sentiments_for_multiple_politicians(engine, ids_for_query)
            if not df_scores_raw.empty:
                # Calculate median sentiment score for each politician
                median_scores = df_scores_raw.groupby('politician_name')['sentiment_score'].median()
                # Get politician names sorted by median score in descending order
                sorted_names_by_median = median_scores.sort_values(ascending=False).index.tolist()
                
                # Reorder the 'politician_name' column based on the median score
                df_scores_raw['politician_name'] = pd.Categorical(
                    df_scores_raw['politician_name'], 
                    categories=sorted_names_by_median, 
                    ordered=True
                )
                df_scores_raw.sort_values('politician_name', inplace=True)
                
                histogram_buf = plot_multi_sentiment_histograms_to_image(df_scores_raw, num_bins=12)
                lookup_tab_data['histogram_img_base64'] = get_image_as_base64(histogram_buf)

            top_word_tables_list = []
            for pid in selected_politician_ids:
                pname = politicians_list_df[politicians_list_df['politician_id'] == pid]['name'].iloc[0]
                top_words_df = fetch_top_weekly_word_for_politician(engine, pid)
                top_words_list_for_template = []
                if not top_words_df.empty:
                    top_words_list_for_template = top_words_df.to_dict(orient='records')
                    for item in top_words_list_for_template:
                        if 'Week Start' in item and hasattr(item['Week Start'], 'strftime'):
                            item['Week Start'] = item['Week Start'].strftime('%Y-%m-%d')
                top_word_tables_list.append({
                    'politician_name': pname,
                    'top_words_list': top_words_list_for_template
                })
            lookup_tab_data['top_word_tables'] = top_word_tables_list

    elif active_tab == 'compare':
        politicians_by_votes_df = fetch_sentiment_distribution_per_politician(engine, min_total_votes_threshold=1, sort_by_total_votes=True)
        MAX_HEATMAP_POLITICIANS_CONST = 30
        compare_data_dict = {'MAX_HEATMAP_POLITICIANS': MAX_HEATMAP_POLITICIANS_CONST, 'available_politicians': [], 'selected_politician_ids': [], 'heatmap_img_base64': None, 'comparison_df_html': None, 'error_message': None, 'df_for_comparison_calc': pd.DataFrame()}
        if not SPACY_MODEL_LOADED:
            compare_data_dict['error_message'] = "Semantic comparison calculation is disabled because the required spaCy language model ('en_core_web_md') is not installed."
        else:
            df_all_for_selection = politicians_by_votes_df
            if not df_all_for_selection.empty:
                df_form_list = df_all_for_selection.sort_values('politician_name', ascending=True)
                compare_data_dict['available_politicians'] = df_form_list[['politician_id', 'politician_name']].to_dict(orient='records')
            selected_politician_ids_str = request.args.getlist('politician_ids_compare')
            selected_politician_ids = [int(pid) for pid in selected_politician_ids_str if pid.isdigit()]
            if not selected_politician_ids_str and not df_all_for_selection.empty:
                selected_politician_ids = df_all_for_selection['politician_id'].head(min(5, len(df_all_for_selection))).tolist()
            if "All" in request.args.get('politician_select_mode_compare', '') and not df_all_for_selection.empty:
                ids_for_calc = df_all_for_selection['politician_id'].head(MAX_HEATMAP_POLITICIANS_CONST).tolist()
            else:
                ids_for_calc = selected_politician_ids
            compare_data_dict['selected_politician_ids'] = selected_politician_ids
            if ids_for_calc:
                df_word_counts = fetch_word_counts_per_politician(engine, politician_ids_list=ids_for_calc)
                if not df_word_counts.empty and 'politician_name' in df_word_counts.columns:
                    politician_docs = {name: dict(zip(group['word'], group['count'])) for name, group in df_word_counts.groupby('politician_name')}
                    politician_vectors = {}
                    vector_dim = nlp.vocab.vectors.shape[1]
                    for name, word_counts in politician_docs.items():
                        doc_vector = np.zeros(vector_dim, dtype='float32'); total_weight = 0
                        for word, count in word_counts.items():
                            token = nlp.vocab[word.lower()]
                            if token.has_vector and token.is_alpha and not token.is_stop:
                                doc_vector += token.vector * count; total_weight += count
                        if total_weight > 0: politician_vectors[name] = doc_vector / total_weight
                    valid_politicians = list(politician_vectors.keys())
                    if len(valid_politicians) > 1:
                        names_for_matrix = valid_politicians
                        vectors_for_matrix = np.array([politician_vectors[name] for name in names_for_matrix])
                        comp_matrix = cosine_similarity(vectors_for_matrix)
                        comp_df = pd.DataFrame(comp_matrix, index=names_for_matrix, columns=names_for_matrix)
                        original_name_order = df_all_for_selection[df_all_for_selection['politician_id'].isin(ids_for_calc)]['politician_name'].tolist()
                        ordered_names_in_matrix = [name for name in original_name_order if name in comp_df.index]
                        if ordered_names_in_matrix: comp_df = comp_df.reindex(index=ordered_names_in_matrix, columns=ordered_names_in_matrix)
                        heatmap_buf = plot_comparison_heatmap_to_image(comp_df, title="Semantic Similarity Heatmap", cbar_label="Cosine Similarity of Word Collections (0-1)")
                        compare_data_dict['heatmap_img_base64'] = get_image_as_base64(heatmap_buf)
                        compare_data_dict['comparison_df_html'] = comp_df.style.format("{:.3f}").to_html(classes='styled-table', border=0)
                        compare_data_dict['df_for_comparison_calc'] = comp_df
                    else: compare_data_dict['error_message'] = "Could not generate meaningful semantic profiles for more than one selected politician."
                else: compare_data_dict['error_message'] = "No word data found for the selected politician(s)."
            else:
                if not compare_data_dict.get('error_message'): compare_data_dict['error_message'] = "No politicians selected for comparison analysis."

    elif active_tab == 'overview':
        metrics = fetch_dataset_metrics(engine)
        overview_tab_data = {'metrics': metrics}

    elif active_tab == 'feed':
        daily_activity_df = fetch_daily_activity(engine)
        activity_graph_img_base64 = None
        if not daily_activity_df.empty:
            activity_graph_buf = plot_daily_activity_to_image(daily_activity_df)
            activity_graph_img_base64 = get_image_as_base64(activity_graph_buf)
        
        # Fetch last 100 items regardless of date
        feed_df = fetch_feed_updates(engine, limit=100)
        
        feed_list_for_template = []
        if not feed_df.empty:
            feed_list_for_template = feed_df.to_dict(orient='records')
        feed_data_dict = {
            'activity_graph_img_base64': activity_graph_img_base64,
            'latest_feed_items': feed_list_for_template,
            'feed_display_period_start': None,
            'feed_display_period_end': None
        }

    return render_template('index.html',
                           active_tab=active_tab,
                           lookup_data=lookup_tab_data,
                           compare_data=compare_data_dict,
                           overview_data=overview_tab_data,
                           feed_data=feed_data_dict,
                           engine_available=bool(engine)
                          )
# --- Favicon & 404 Routes ---
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/robots.txt')
def robots_txt():
    return send_from_directory(app.static_folder, 'robots.txt')

@app.errorhandler(404)
def page_not_found(e):
    # Pass engine_available for template, if base.html expects it
    return render_template('404.html', engine_available=bool(engine)), 404

@app.route('/healthz')
def health_check():
    return "OK", 200

# --- Main Entry Point ---
if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    if not debug_mode and DEPLOY_ENV == 'DEVELOPMENT': # This seems like a conflicting logic, removed for simplicity
        debug_mode = True
        app.logger.info("Forcing debug mode ON for local 'python3 app.py' execution as FLASK_ENV was not 'development'.")
    port_num = int(os.environ.get('PORT', 5001))
    app.logger.info(f"Attempting to run app with debug_mode: {debug_mode} on port {port_num}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port_num)