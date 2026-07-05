import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import Select 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC 
from webdriver_manager.chrome import ChromeDriverManager 
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, TimeoutException

# --- Configuration for Brookings.edu ---
KEYWORD = "taiwan"
START_URL = f"https://www.brookings.edu/?s={KEYWORD}"
# CRITICAL CHANGE: Output file name updated to reflect scraping all articles
OUTPUT_FILENAME = f"{KEYWORD.capitalize()}_Brookings_Selenium_Dynamic_V_ALL.csv" 

# --- CRITICALLY REVISED SELECTORS ---
# These selectors are robust fallback options for finding article containers
ARTICLE_CONTAINER_SELECTORS = [
    "main article",         
    "article.c-card",
    "div.search-results-list article",
    "section.content-list article",
    "div.search-result-items article"
]

# Selectors relative to the successfully identified ARTICLE_CONTAINER:
# FIX: Removed previous CSS selectors and defined new, reliable RELATIVE XPATHS
DATE_XPATH_SEARCH_REL = "./div[2]/span[2]/span[2]/p[2]"
AUTHORS_XPATH_SEARCH_REL = "./div[2]/span[2]/span[2]/p[1]"
LABELS_SELECTOR = "div.content-type-wrapper span.content-type-label" # Included for completeness
SUMMARY_SELECTOR = "p.excerpt" # Included for completeness

# Article Page XPaths (Used inside get_article_full_text)
AUTHOR_XPATH_ARTICLE = "/html/body/div[4]/main/section[1]/div[1]/div[2]/div/div[1]/div/div[2]/h5/div/a"
DATE_XPATH_ARTICLE = "/html/body/div[4]/main/section[1]/div[1]/div[2]/div/div[1]/div/div[2]/div/p"

# NEW: More robust CSS selectors for standard Brookings article header elements
AUTHOR_CSS_ROBUST = "div.by-line a"
DATE_CSS_ROBUST = "div.by-line time"

# NEW: Selectors for max articles per page dropdown (based on user-provided path)
ARTICLES_PER_PAGE_SELECT_XPATH = "/html/body/div[4]/main/section/div[2]/div/div[6]/div/div/div[2]/div[2]/div[3]/div/div/select"
ARTICLES_PER_PAGE_VALUE = "40" # The value to select (highest initial load)
# --- CRITICAL NEW SELECTOR FOR PAGINATION ---
SHOW_MORE_BUTTON_XPATH = "/html/body/div[4]/main/section/div[2]/div/div[6]/div/div/div[2]/div[2]/div[2]/div/div/button/div/div" 

# Title selectors
TITLE_LINK_XPATH = "./a" 
TITLE_SPAN_CSS = "span.sr-only" 

NEXT_PAGE_SELECTOR = "a.next" 
# -----------------------------------------------------------------

def setup_driver():
    """Sets up the Selenium Chrome driver in headless mode."""
    chrome_options = Options()
    chrome_options.add_argument("--headless") 
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

def get_article_full_text(driver, link):
    """
    Opens a new tab to scrape the full text, author, and date from the individual article page.
    This version includes robust fallbacks for author/date extraction.
    Returns: full_text, article_author, article_date
    """
    full_text = ""
    article_author = "Unknown"
    article_date = None
    original_window = driver.current_window_handle
    
    # Check if the URL is pointing to a different domain
    if not "brookings.edu" in link:
        print(f"Skipping detailed scrape for external link: {link}")
        return full_text, article_author, article_date

    try:
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[-1]) # Switch to the new tab
        
        driver.set_page_load_timeout(15) 
        driver.get(link)
        time.sleep(3) # Simple wait for content to load

        # 1. Full Text Extraction (targets common body paragraphs)
        content_paragraphs = driver.find_elements(By.CSS_SELECTOR, "div.post-body p")
        full_text = "\n".join([p.text for p in content_paragraphs if p.text.strip() != ""])

        # 2. Robust Author Extraction (CSS Selector first)
        try:
            author_element = driver.find_element(By.CSS_SELECTOR, AUTHOR_CSS_ROBUST)
            article_author = author_element.text.strip()
        except NoSuchElementException:
            # Fallback to the brittle XPath
            try:
                author_element = driver.find_element(By.XPATH, AUTHOR_XPATH_ARTICLE)
                article_author = author_element.text.strip()
            except NoSuchElementException:
                pass
            
        # 3. Robust Date Extraction (CSS Selector first)
        try:
            date_element = driver.find_element(By.CSS_SELECTOR, DATE_CSS_ROBUST)
            # Use get_attribute('datetime') for cleaner date/time if available, otherwise use text
            article_date = date_element.get_attribute('datetime') or date_element.text.strip()
        except NoSuchElementException:
            # Fallback to the brittle XPath
            try:
                date_element = driver.find_element(By.XPATH, DATE_XPATH_ARTICLE)
                article_date = date_element.text.strip()
            except NoSuchElementException:
                pass


    except TimeoutException:
        print(f"Navigation timed out for article: {link}")
    except Exception as e:
        # Catch other unexpected errors like connection issues
        # print(f"Error during article page data fetch: {e}") 
        pass
    finally:
        # Ensure the new tab is closed and we switch back to the original tab
        if driver.current_window_handle != original_window:
            driver.close()
        driver.switch_to.window(original_window)
        driver.set_page_load_timeout(-1) # Reset timeout
    
    return full_text, article_author, article_date

def extract_title_and_link(article):
    """
    Uses relative XPath/CSS to find the direct anchor tag and its nested hidden span text.
    """
    link = None
    title = None
    
    try:
        # 1. Find the anchor element (<a>) using relative XPath from the article container
        link_element = article.find_element(By.XPATH, TITLE_LINK_XPATH) 
        link = link_element.get_attribute("href")
        
        # 2. Find the nested span with the actual hidden title text 
        try:
            # Search for the span.sr-only inside the link element
            title_span = link_element.find_element(By.CSS_SELECTOR, TITLE_SPAN_CSS)
            title = title_span.text.strip()
        except NoSuchElementException:
            # Fallback: If the hidden span is missing, use the link's entire visible text content
            title = link_element.text.strip()
            
    except NoSuchElementException:
        # If the direct link is missing, we return None, None to skip the record.
        return None, None
    except Exception:
        # Catch other unexpected extraction errors
        return None, None
    
    # CRITICAL CHECK: ensure the title is not empty before returning
    if not title:
        return None, None
        
    return title, link

def find_working_selector(driver):
    """Iterates through candidate selectors to find one that returns articles."""
    for selector in ARTICLE_CONTAINER_SELECTORS:
        articles = driver.find_elements(By.CSS_SELECTOR, selector)
        if articles:
            print(f"✅ Found {len(articles)} articles using container selector: '{selector}'")
            return selector
    return None

def scrape_brookings_articles():
    driver = setup_driver()
    driver.get(START_URL) 
    time.sleep(5) # Initial wait for page load and JavaScript execution
    
    print(f"✅ Driver successfully navigated to search URL: {START_URL}")

    data = []
    
    # --- FIND THE WORKING CONTAINER SELECTOR ON PAGE 1 ---
    working_article_selector = find_working_selector(driver)
    
    if not working_article_selector:
        print(f"Fatal Error: None of the backup selectors found any articles on the page. Please inspect the page manually.")
        driver.quit()
        return
    
    # --- Select Max Articles per Page (Set to 40) ---
    try:
        wait = WebDriverWait(driver, 10)
        select_element = wait.until(
            EC.presence_of_element_located((By.XPATH, ARTICLES_PER_PAGE_SELECT_XPATH))
        )
        select = Select(select_element)
        print(f"\nAttempting to set articles per page to {ARTICLES_PER_PAGE_VALUE}...")
        select.select_by_visible_text(ARTICLES_PER_PAGE_VALUE)
        time.sleep(5) 
        print(f"✅ Successfully set articles per page to {ARTICLES_PER_PAGE_VALUE}.")
        
        # Re-find the working selector since the page content has refreshed
        working_article_selector = find_working_selector(driver)
        if not working_article_selector:
            print("Fatal Error: Could not find articles after changing articles per page setting.")
            driver.quit()
            return

    except Exception as e:
        print(f"Warning: Articles per page dropdown failed ({e}). Proceeding with default settings.")
    # ------------------------------------------

    print(f"\nStarting crawl for keyword: '{KEYWORD}' on Brookings.edu, targeting ALL records...")

    # --- Pagination Loop: Use "Show More" until it fails ---
    while True:
        
        # --- 1. Scrape Articles Currently Visible ---
        article_containers = driver.find_elements(By.CSS_SELECTOR, working_article_selector)
        current_data_count = len(data) # The index where we left off
        
        # Calculate how many items we expect to be processed in this full batch.
        items_expected_to_be_processed = len(article_containers)
        # If this is the very first iteration (current_data_count == 0), we skip one article.
        if current_data_count == 0:
            items_expected_to_be_processed = len(article_containers) - 1

        # Iterate over new articles only
        for i in range(current_data_count, len(article_containers)):
                
            article = article_containers[i]
            
            # --- START: SKIP LOGIC (Only applies to the first container on the first load) ---
            if i == 0 and current_data_count == 0:
                print("⏭️ Skipping the first article as requested.")
                continue
            # --- END: SKIP LOGIC ---
            
            title = None
            link = None
            authors = "Unknown"
            date = None
            full_text = ""

            try:
                # --- 1. TITLE AND LINK EXTRACTION (Search Page) ---
                title, link = extract_title_and_link(article)
                if not title or not link:
                    continue
                
                # --- 2. DATE EXTRACTION (Search Page - FIXED WITH RELATIVE XPATH) ---
                try:
                    # Uses the relative XPATH derived from user's input: ./div[2]/span[2]/span[2]/p[2]
                    date_element = article.find_element(By.XPATH, DATE_XPATH_SEARCH_REL)
                    date = date_element.text.strip() # Assuming text is clean date string
                except NoSuchElementException:
                    pass

                # --- 3. AUTHOR EXTRACTION (Search Page - FIXED WITH RELATIVE XPATH) ---
                try:
                    # Uses the relative XPATH derived from user's input: ./div[2]/span[2]/span[2]/p[1]
                    author_element = article.find_element(By.XPATH, AUTHORS_XPATH_SEARCH_REL)
                    authors = author_element.text.strip()
                except NoSuchElementException:
                    pass
                
                
                # --- 4. FULL TEXT FETCH (Requires new tab navigation) ---
                full_text_fetched, article_author_fallback, article_date_fallback = get_article_full_text(driver, link)
                
                full_text = full_text_fetched
                # Fallback check: use the article page data if the search page data was poor
                authors = article_author_fallback if article_author_fallback != "Unknown" else authors
                date = article_date_fallback if article_date_fallback else date


                print(f"COLLECTED {len(data) + 1} | TITLE: '{title}' | Author: {authors} | Date (Search Page): {date}")
                
                data.append({
                    "Title": title,
                    "Authors": authors,
                    "Date": date,
                    "URL": link,
                    "Full Text": full_text
                })

            except Exception as e:
                # Note: 'i' is the index within the currently visible articles, 
                # but it helps identify where the failure occurred.
                print(f"Unhandled error processing article {i+1}: {e}. Skipping.")
                continue 

        # --- 2. Attempt to Click "Show More" ---
        
        # FIX: Check if we have successfully collected the number of items we expected to process.
        if len(data) == items_expected_to_be_processed:
            try:
                print(f"\n--- Current count: {len(data)}. Attempting to click 'Show More'...")
                
                # Wait for the button to be clickable
                wait = WebDriverWait(driver, 10) 
                show_more_button = wait.until(
                    EC.element_to_be_clickable((By.XPATH, SHOW_MORE_BUTTON_XPATH))
                )
                
                # Scroll to the button to ensure it's in the viewport before clicking
                driver.execute_script("arguments[0].scrollIntoView(true);", show_more_button)
                time.sleep(1) # Small pause after scrolling
                
                show_more_button.click()
                time.sleep(5) # Wait for the new content to load (AJAX update)
                
                print("✅ Successfully clicked 'Show More'. New articles should be visible.")

            except TimeoutException:
                print("\n🛑 Timeout: 'Show More' button not found or clicked. Assuming all results have been loaded. Stopping crawl.")
                break
            except Exception as e:
                print(f"\n🛑 Error clicking 'Show More': {e}. Stopping crawl.")
                break
        else:
            # This case means we processed some, but not all, of the currently visible containers.
            # This suggests a permanent issue with one of the articles that caused an unhandled error/skip
            # and prevents the successful collection of the full batch.
            print(f"\n🛑 Processed {len(data)} articles, but expected {items_expected_to_be_processed} from the visible list. Incomplete batch detected. Stopping crawl.")
            break


    driver.quit()
    df = pd.DataFrame(data)
    
    # Save the dataframe to a CSV file
    df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8-sig')
    print(f"\n=======================================================")
    print(f"Scraping complete. Collected **{len(df)}** records.")
    print(f"Data saved to **{OUTPUT_FILENAME}**")
    print(f"=======================================================")

if __name__ == "__main__":
    scrape_brookings_articles()