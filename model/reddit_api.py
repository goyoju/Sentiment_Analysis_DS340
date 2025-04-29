import praw
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Reddit API credentials
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'SentimentAnalyzer/1.0')

def get_reddit_instance():
    """Create and return a Reddit API instance"""
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        raise ValueError("Reddit API credentials not found. Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.")
    
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

def fetch_reddit_comments(keyword, limit=100, subreddits=None):
    """
    Fetch comments from Reddit that contain the specified keyword
    
    Args:
        keyword (str): The keyword to search for
        limit (int): Maximum number of comments to fetch
        subreddits (list): List of subreddits to search in. If None, searches all of Reddit
        
    Returns:
        list: A list of comment texts containing the keyword
    """
    try:
        reddit = get_reddit_instance()
        comments = []
        
        # If no specific subreddits are provided, search in r/all
        if not subreddits:
            subreddits = ['all']
        
        for subreddit_name in subreddits:
            subreddit = reddit.subreddit(subreddit_name)
            
            # Search for submissions containing the keyword
            for submission in subreddit.search(keyword, limit=25, sort='hot'):
                # Expand all comments - this can be resource-intensive for large threads
                submission.comments.replace_more(limit=0)
                
                # Process all comments in the submission
                for comment in submission.comments.list():
                    if keyword.lower() in comment.body.lower():
                        # Clean the comment text (remove newlines, excessive spaces)
                        clean_comment = ' '.join(comment.body.split())
                        comments.append(clean_comment)
                        
                        # Check if we've reached the limit
                        if len(comments) >= limit:
                            return comments
                
                # Reddit has rate limits, so we should be respectful
                time.sleep(0.1)
        
        return comments
    
    except Exception as e:
        print(f"Error fetching comments from Reddit: {e}")
        return []

def fetch_sentiment_trend(keyword, timeframes=None):
    """
    Fetch comments from different timeframes to analyze sentiment trends over time
    
    Args:
        keyword (str): The keyword to search for
        timeframes (list): List of timeframes to search (day, week, month, year)
        
    Returns:
        dict: Sentiment data organized by timeframe
    """
    if not timeframes:
        timeframes = ['day', 'week', 'month', 'year']
    
    reddit = get_reddit_instance()
    results = {}
    
    for timeframe in timeframes:
        comments = []
        
        # Search posts from the specified timeframe
        for submission in reddit.subreddit('all').search(
            keyword, 
            limit=10, 
            sort='hot', 
            time_filter=timeframe
        ):
            submission.comments.replace_more(limit=0)
            
            for comment in submission.comments.list():
                if keyword.lower() in comment.body.lower():
                    clean_comment = ' '.join(comment.body.split())
                    comments.append(clean_comment)
            
            # Reddit has rate limits
            time.sleep(0.1)
        
        results[timeframe] = comments
    
    return results

if __name__ == "__main__":
    # Test the functions
    keyword = "bitcoin"
    comments = fetch_reddit_comments(keyword, limit=5)
    print(f"Found {len(comments)} comments containing '{keyword}':")
    for i, comment in enumerate(comments[:5]):
        print(f"{i+1}. {comment[:100]}...")