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
# CRITICAL CHANGE: Output file name updated for ALL records
OUTPUT_FILENAME = f"{KEYWORD.capitalize()}_Brookings_Selenium_Dynamic_V_ALL_FULL.csv" 
# MAX_RECORDS has been removed

# --- CRITICALLY REVISED SELECTORS ---
# These selectors are robust fallback options for finding article containers
ARTICLE_CONTAINER_SELECTORS = [
    "main article",         
    "article.c-card",
    "div.search-results-list article",
    "section.content-list article",
    "div.search-result-items article"
]

# Selectors relative to the successfully identified ARTICLE_CONTAINER (Used for Author/Date/Title):
TITLE_LINK_XPATH = "./a" 
# Specific relative XPATHS derived from user input for search page extraction
TITLE_XPATH_SEARCH_REL = "./div[2]/span[2]/span[1]/span/span" 
AUTHORS_XPATH_SEARCH_REL = "./div[2]/span[2]/span[2]/p[1]"
DATE_XPATH_SEARCH_REL = "./div[2]/span[2]/span[2]/p[2]" 

# Article Content XPaths (Used inside get_article_full_text for content comparison)
FULL_TEXT_XPATHS_PRIMARY = [
    "/html/body/div[4]/main/section[2]/div", # General container for primary content
    "/html/body/div[1]/div[2]/div/main/div/article/div/div[1]/div[1]/div"  # Pop-up/Modal content area
]
# Fallback CSS selectors to capture all paragraph content within the main post body
FULL_TEXT_CSS_FALLBACK = ["div.post-body p", "article.post-body p", "div.post-body-container p"] 

# Article Page XPaths (Fallbacks used inside get_article_full_text for header info)
AUTHOR_XPATH_ARTICLE = "/html/body/div[4]/main/section[1]/div[1]/div[2]/div/div[1]/div/div[2]/h5/div/a"
DATE_XPATH_ARTICLE = "/html/body/div[4]/main/section[1]/div[1]/div[2]/div/div[1]/div/div[2]/div/p"

# NEW: More robust CSS selectors for standard Brookings article header elements
AUTHOR_CSS_ROBUST = "div.by-line a"
DATE_CSS_ROBUST = "div.by-line time"

# NEW: Selectors for max articles per page dropdown (based on user-provided path)
ARTICLES_PER_PAGE_SELECT_XPATH = "/html/body/div[4]/main/section/div[2]/div/div[6]/div/div/div[2]/div[2]/div[3]/div/div/select"
ARTICLES_PER_PAGE_VALUE = "40" # The value to select (highest initial load)

# Pagination Selector (The Show More Button)
SHOW_MORE_BUTTON_XPATH = "/html/body/div[4]/main/section/div[2]/div/div[6]/div/div/div[2]/div[2]/div[2]/div/div/button/div/div" 
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
    Implements comparison logic to find the longest content from specified XPaths.
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
        
        driver.set_page_load_timeout(20) 
        driver.get(link)
        time.sleep(3) # Simple wait for content to load

        # 1. Full Text Extraction (Comparison Logic)
        best_full_text = ""
        
        # Paths to check: Primary XPATH containers
        all_paths_to_check = FULL_TEXT_XPATHS_PRIMARY 
        
        for path in all_paths_to_check:
            current_text = ""
            try:
                # Use XPATH for the specific containers provided by the user
                container_element = driver.find_element(By.XPATH, path)
                
                # Retrieve content from all p tags within the container (robust)
                content_paragraphs = container_element.find_elements(By.TAG_NAME, "p")
                
                if content_paragraphs:
                    text_content = [p.text.strip() for p in content_paragraphs]
                    current_text = "\n".join([t for t in text_content if t])
                
                # Comparison logic: Keep the longest text found so far
                if len(current_text) > len(best_full_text):
                    best_full_text = current_text
                    
            except Exception:
                continue # Try the next XPath/selector if this one fails
        
        # Fallback to general CSS if the specific XPATHS failed or returned short text
        if not best_full_text or len(best_full_text) < 500: # Arbitrary minimum length check
             try:
                # Use a very general CSS selector to find common post body paragraphs
                content_paragraphs_fallback = driver.find_elements(By.CSS_SELECTOR, "div.post-body p")
                if content_paragraphs_fallback:
                    text_content_fallback = [p.text.strip() for p in content_paragraphs_fallback]
                    fallback_text = "\n".join([t for t in text_content_fallback if t])
                    if len(fallback_text) > len(best_full_text):
                        best_full_text = fallback_text
             except Exception:
                 pass
                 
        full_text = best_full_text

        # 2. Robust Author Extraction (CSS Selector first)
        try:
            author_element = driver.find_element(By.CSS_SELECTOR, AUTHOR_CSS_ROBUST)
            article_author = author_element.text.strip()
        except NoSuchElementException:
            try:
                author_element = driver.find_element(By.XPATH, AUTHOR_XPATH_ARTICLE)
                article_author = author_element.text.strip()
            except NoSuchElementException:
                pass
            
        # 3. Robust Date Extraction (CSS Selector first)
        try:
            date_element = driver.find_element(By.CSS_SELECTOR, DATE_CSS_ROBUST)
            article_date = date_element.get_attribute('datetime') or date_element.text.strip()
        except NoSuchElementException:
            try:
                date_element = driver.find_element(By.XPATH, DATE_XPATH_ARTICLE)
                article_date = date_element.text.strip()
            except NoSuchElementException:
                pass


    except TimeoutException:
        print(f"Navigation timed out for article: {link}")
    except Exception as e:
        print(f"Error during article page data fetch for {link}: {e}") 
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
    Uses relative XPATH to find the title text and the link.
    """
    link = None
    title = None
    
    try:
        # 1. Find the anchor element (<a>) using the container link path
        link_element = article.find_element(By.XPATH, TITLE_LINK_XPATH) 
        link = link_element.get_attribute("href")
        
        # 2. Find the nested span with the actual title text using the specific relative XPath
        try:
            # Use article.find_element to search relative to the container
            title_span = article.find_element(By.XPATH, TITLE_XPATH_SEARCH_REL) 
            title = title_span.text.strip()
        except NoSuchElementException:
            # Fallback to the link's entire visible text content if the complex span fails
            title = link_element.text.strip()
            
    except NoSuchElementException:
        return None, None
    except Exception:
        return None, None
    
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
        
        # Select the option by its visible text '40'
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

    print(f"\nStarting crawl for keyword: '{KEYWORD}' on Brookings.edu, targeting ALL records (FULL TEXT MODE)...")

    # --- Pagination Loop: Use "Show More" until it fails ---
    while True:
        
        # --- 1. Scrape Articles Currently Visible ---
        article_containers = driver.find_elements(By.CSS_SELECTOR, working_article_selector)
        current_data_count = len(data) # The index where we left off
        
        # Calculate how many items we expect to be processed in this full batch.
        items_expected_to_be_processed = len(article_containers)
        # If this is the very first iteration (current_data_count == 0), we skip one article.
        initial_skip = False
        if current_data_count == 0:
            items_expected_to_be_processed = len(article_containers) - 1
            initial_skip = True

        # Track number of successful scrapes in this batch
        successful_scrapes_in_batch = 0
            
        # Iterate over new articles only
        for i in range(current_data_count, len(article_containers)):
                
            article = article_containers[i]
            
            # --- START: SKIP LOGIC 1 (Initial Skip) ---
            if i == 0 and initial_skip:
                print("⏭️ Skipping the first article as requested.")
                continue
            # --- END: SKIP LOGIC 1 ---
            
            title = None
            link = None
            authors = "Unknown"
            date = None
            full_text = ""

            try:
                # --- 1. TITLE AND LINK EXTRACTION (Search Page) ---
                title, link = extract_title_and_link(article)
                if not title or not link:
                    # Treat missing title/link as a soft skip for this container
                    items_expected_to_be_processed -= 1
                    continue
                
                # --- 2. EVENT URL SKIP ---
                if link and link.startswith("https://www.brookings.edu/events"):
                    print(f"⏭️ Skipping record (Event URL): {link}")
                    items_expected_to_be_processed -= 1 
                    continue
                
                # --- 3. SEARCH PAGE EXTRACTION (Author/Date via Relative XPATH) ---
                try:
                    date_element = article.find_element(By.XPATH, DATE_XPATH_SEARCH_REL)
                    date = date_element.text.strip() 
                except NoSuchElementException:
                    pass

                try:
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


                print(f"COLLECTED {len(data) + 1} | TITLE: '{title}' | Author: {authors} | Date: {date}")
                
                data.append({
                    "Title": title,
                    "Authors": authors,
                    "Date": date,
                    "URL": link,
                    "Full Text": full_text 
                })
                
                successful_scrapes_in_batch += 1 # Increment success counter

            except Exception as e:
                print(f"Unhandled error processing article {i+1}: {e}. Skipping.")
                # We skip, so we must decrease the count of expected successful scrapes
                items_expected_to_be_processed -= 1
                continue 

        # --- 2. Attempt to Click "Show More" ---
        
        # Check if we successfully collected the expected number of items (within a tolerance for unexpected final batch size).
        # We rely on the button existence as the ultimate stop condition.
        if successful_scrapes_in_batch > 0:
            
            try:
                # If the last scraped article index is the last visible container index, we should click "Show More"
                # If we've processed all visible containers, attempt the click.
                
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
                
                # Check for zero articles loaded in the next batch (meaning end of results even if button exists)
                if len(driver.find_elements(By.CSS_SELECTOR, working_article_selector)) == len(article_containers):
                    print("🛑 'Show More' clicked, but no new articles loaded. Assuming end of results.")
                    break


            except TimeoutException:
                print("\n🛑 Timeout: 'Show More' button not found or clicked. Assuming all results have been loaded. Stopping crawl.")
                break
            except Exception as e:
                print(f"\n🛑 Error clicking 'Show More': {e}. Stopping crawl.")
                break
        else:
            # If successful_scrapes_in_batch is 0, it means either:
            # a) No more articles are visible (end of results).
            # b) The only visible articles were skipped or failed.
            print(f"\n🛑 No new articles were successfully scraped in this batch. Assuming end of results. Stopping crawl.")
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