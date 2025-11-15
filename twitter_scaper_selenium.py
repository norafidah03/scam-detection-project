from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
import time
import pandas as pd
import os

# --- Configuration ---
CHROMEDRIVER_PATH = "C:/Development/venv_py310/Scripts/chromedriver.exe"
output_filename = "C:/Development/malaysian_scams_tweets_selenium.csv"

# Search query
SEARCH_QUERY = '"duit mudah" AND "wasap" OR "whatsapp"' # Change to search for other keywords
SCROLL_PAUSES = 50 # Number of times to scroll down to load more tweets
SCROLL_DELAY = 10 # Seconds to wait after each scroll

# --- Setup WebDriver ---
try:
    service = Service(executable_path=CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service)
    driver.maximize_window()
    print("WebDriver initialized successfully.")
except Exception as e:
    print(f"Error initializing WebDriver: {e}")
    print("Please ensure ChromeDriver is in the specified path and matches your Chrome browser version.")
    exit() # Exit if driver fails to initialize

# --- Navigate to X and Search ---
try:
    driver.get("https://twitter.com/search?q=&src=typed_query") # Direct link to search
    print(f"Navigated to X search page.")
    time.sleep(30) # Wait for page to load

    try:
        search_input = driver.find_element(By.CSS_SELECTOR, "input[data-testid='SearchBox_Search_Input']")
    except:
        print("Could not find the initial search input. Trying alternative selector...")
        search_input = driver.find_element(By.XPATH, "//input[@aria-label='Search query']")


    search_input.send_keys(SEARCH_QUERY)
    search_input.send_keys(Keys.ENTER)
    print(f"Searched for: '{SEARCH_QUERY}'")
    time.sleep(5) # Wait for search results to load

    # --- Initialize data collection ---
    collected_tweets_data = []
    # Use a set to store unique tweet IDs to prevent duplicates
    collected_tweet_ids = set()

    # --- Scroll to Load More Tweets and Extract Incrementally ---
    last_height = driver.execute_script("return document.body.scrollHeight")
    for i in range(SCROLL_PAUSES):
        print(f"Scrolling down... ({i+1}/{SCROLL_PAUSES})")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_DELAY) # Wait for new content to load

        # --- Extract Tweets from the current view ---
        current_page_tweet_cells = driver.find_elements(By.XPATH, "//div[@data-testid='cellInnerDiv']")
        print(f"  Found {len(current_page_tweet_cells)} tweet cells on current view in this scroll.")

        for cell_el in current_page_tweet_cells:
            tweet_id = None # Initialize tweet_id for each cell
            try:
                # 1. Try to get data-tweet-id attribute first
                tweet_id = cell_el.get_attribute("data-tweet-id")
                print(f"Debug: Attempted data-tweet-id for a cell: {tweet_id}")

                # 2. If data-tweet-id is not found, try extracting from permalink URL
                if not tweet_id:
                    try:
                        permalink_el = cell_el.find_element(By.XPATH, ".//a[contains(@href, '/status/') and @role='link']")
                        tweet_url = permalink_el.get_attribute("href")

                        if tweet_url:
                            # Extract the last part of the URL, which is typically the tweet ID
                            tweet_id = tweet_url.split('/')[-1]
                            print(f"Debug: Tweet ID from URL: {tweet_id}")
                    except Exception as url_e:
                        print(f"Debug: Could not find tweet ID from permalink for this cell: {url_e}")
                        pass # tweet_id remains None if URL extraction fails

                if tweet_id and tweet_id in collected_tweet_ids:
                    # print(f"  Skipping duplicate tweet ID: {tweet_id}")
                    continue # Skip if already collected

                # Find the tweet text within the current cell
                tweet_text_div = cell_el.find_element(By.XPATH, ".//div[@data-testid='tweetText']")
                tweet_text = tweet_text_div.text

                # --- OPTIONAL: Extract other data ---
                username = "N/A"
                timestamp = "N/A"
                try:
                    # The username is typically in a div with data-testid="User-Name"
                    # Might need to adjust this depending on the exact HTML
                    # Look for the span with the actual username text
                    username_el = cell_el.find_element(By.XPATH, ".//div[@data-testid='User-Name']//span[contains(@class, 'r-1awozwy')]")
                    username = username_el.text
                except:
                    pass # Keep as N/A if not found

                try:
                    # The time element usually has a time tag with datetime attribute
                    time_el = cell_el.find_element(By.TAG_NAME, "time")
                    timestamp = time_el.get_attribute("datetime")
                except:
                    pass # Keep as N/A if not found

                collected_tweets_data.append({
                    'tweet_id': tweet_id, # Add tweet ID for reference
                    'tweet_text': tweet_text,
                    'username': username,
                    'timestamp': timestamp
                })
                if tweet_id: # Add ID to set only if it's not None
                    collected_tweet_ids.add(tweet_id)

            except Exception as e:
                # This cell might not be a valid tweet (e.g., an ad or suggestion), or extraction failed
                # print(f"Skipping a cell (not a tweet or error extracting text: {e})")
                continue # Skip this cell and move to the next one

        # Check for end of page after collecting tweets from current view
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            print("End of page reached or no more content loaded.")
            break
        last_height = new_height

    print("Finished scrolling and collection.")
    print(f"Total unique tweets collected: {len(collected_tweets_data)}")

    # --- Save Collected Data to CSV ---
    if collected_tweets_data:
        df = pd.DataFrame(collected_tweets_data)

        # Check if the file already exists to decide whether to write the header
        if not os.path.isfile(output_filename):
            # File does not exist, write with header
            df.to_csv(output_filename, mode='a', index=False, encoding='utf-8', header=True)
        else:
            # File exists, append without header
            df.to_csv(output_filename, mode='a', index=False, encoding='utf-8', header=False)

        print(f"Collected {len(df)} tweets and appended to {output_filename}")
    else:
        print("No tweet elements found or extracted after filtering cells.")

except Exception as e:
    # This will print the full traceback for any unhandled errors
    raise e
    # print(f"An error occurred during scraping: {e}") # This line won't be reached if 'raise e' is used

finally:
    # Always close the browser
    if 'driver' in locals() and driver:
        driver.quit()
        print("Browser closed.")

