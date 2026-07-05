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

# --- Configuration ---
KEYWORD = "taiwan"
START_URL = f"https://www.brookings.edu/?s={KEYWORD}"
OUTPUT_FILENAME = f"{KEYWORD.capitalize()}_Brookings_External_Content_FIRST_2.csv" 
DOMAIN_ROOT = "https://www.brookings.edu" # Changed to exclude trailing slash for cleaner matching
MAX_EXTERNAL_TO_SCRAPE = 2 # CRITICAL: Limit to the first 2 external sites found.

# --- SELECTORS ---
# Search Page Containers
ARTICLE_CONTAINER_SELECTORS = ["main article", "article.c-card", "div.search-results-list article"]

# Search Page Metadata (Using the latest robust XPATHS)
# The main link container is the article link itself:
TITLE_LINK_XPATH = "./a" 
TITLE_XPATH_SEARCH_REL = "./div[2]/span[2]/span[1]/span/span" 
AUTHORS_XPATH_SEARCH_REL = "./div[2]/span[2]/span[2]/p[1]"
DATE_XPATH_SEARCH_REL = "./div[2]/span[2]/span[2]/p[2]" 

# Dropdown and Pagination
ARTICLES_PER_PAGE_SELECT_XPATH = "/html/body/div[4]/main/section/div[2]/div/div[6]/div/div/div[2]/div[2]/div[3]/div/div/select"
ARTICLES_PER_PAGE_VALUE = "40" 
SHOW_MORE_BUTTON_XPATH = "/html/body/div[4]/main/section/div[2]/div/div[6]/div/div/div[2]/div[2]/div[2]/div/div/button/div/div" 

# External Content XPaths (CRITICALLY UPDATED based on user input)
# We use leading dots (./) for relative XPATHS if searching from a container, 
# but for maximum flexibility when jumping to an external site, we use global XPATHS (//).
EXTERNAL_CONTENT_XPATHS = [
    "//div[1]/div[2]/div/main/div/article/div/div[1]/div[1]/div", # Pop-up/Modal content area
    "//div[7]/ul/li[1]/div[1]", 
    "//div[1]/div/div/div/main/article/div[1]",
    "//div[@class='post-body']", 
    "//article[@id='main-content']",
]
# -----------------------------------------------------------------

def setup_driver():
    """Sets up the Selenium Chrome driver with anti-bot arguments."""
    chrome_options = Options()
    chrome_options.add_argument("--headless") 
    chrome_options.add_argument("--start-maximized")
    
    # --- Anti-Bot Fixes ---
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    # ----------------------
    
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

def get_external_article_content(driver, link):
    """
    Opens a new tab to scrape the full text from the external article page, 
    checking multiple XPaths and returning the longest text found.
    
    Returns: full_text
    """
    full_text = ""
    original_window = driver.current_window_handle
    best_full_text = ""
    
    print(f"   Navigating to external link: {link}")

    try:
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[-1]) # Switch to the new tab
        
        driver.set_page_load_timeout(30) # Increased timeout for external sites
        driver.get(link)
        time.sleep(5) # Increased initial wait time to help load resources/bypass basic checks

        # --- 1. CAPTCHA/Anti-Bot Check ---
        # Look for common Cloudflare/reCAPTCHA text, suggesting we've been blocked
        if "Verifying you are human" in driver.page_source or "captcha" in driver.current_url:
            print("   ⚠️ Detected CAPTCHA/Anti-Bot verification page. Skipping content fetch.")
            return full_text
        
        # --- 2. Content Extraction ---
        for path in EXTERNAL_CONTENT_XPATHS:
            current_text = ""
            try:
                # Find the container using the specific XPath
                content_container = driver.find_element(By.XPATH, path)
                
                # Extract text from all paragraphs (<p> tags) within the container
                # Note: We specifically search for 'p' tags inside the found container.
                content_paragraphs = content_container.find_elements(By.TAG_NAME, "p")
                current_text = "\n".join([p.text.strip() for p in content_paragraphs if p.text.strip() != ""])

                # Keep the longest text found so far
                if len(current_text) > len(best_full_text):
                    best_full_text = current_text
                    
            except NoSuchElementException:
                continue # Try the next XPath if this one fails

        # Finalize full_text
        if len(best_full_text) > 500: # Requires a minimum length to be a valid article
             full_text = best_full_text
        else:
            print(f"   ⚠️ Content found was too short ({len(best_full_text)} chars). Potential content fetch failure or snippet only.")
        
    except TimeoutException:
        print(f"   ❌ Navigation timed out for external article: {link}")
    except Exception as e:
        print(f"   ❌ Error during external content fetch for {link}: {e}") 
        
    finally:
        if driver.current_window_handle != original_window:
            driver.close()
        driver.switch_to.window(original_window)
        driver.set_page_load_timeout(-1)
    
    return full_text

def find_working_selector(driver):
    """Iterates through candidate selectors to find one that returns articles."""
    for selector in ARTICLE_CONTAINER_SELECTORS:
        articles = driver.find_elements(By.CSS_SELECTOR, selector)
        if articles:
            return selector
    return None

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
            # We use the specific path to get the clean title text
            title_span = article.find_element(By.XPATH, TITLE_XPATH_SEARCH_REL) 
            title = title_span.text.strip()
        except NoSuchElementException:
            # Fallback to the link's entire visible text content 
            title = link_element.text.strip()
            
    except NoSuchElementException:
        return None, None
    except Exception:
        return None, None
    
    if not title:
        return None, None
        
    return title, link

def scrape_external_content():
    driver = setup_driver()
    driver.get(START_URL) 
    time.sleep(5) 
    
    print(f"✅ Driver successfully navigated to search URL: {START_URL}")

    data = []
    last_processed_index = 0 
    
    # --- 1. INITIAL SETUP: Find Selector and Set Page Size to 40 ---
    working_article_selector = find_working_selector(driver)
    if not working_article_selector:
        print(f"Fatal Error: Could not find any article containers.")
        driver.quit()
        return

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
    except Exception as e:
        print(f"Warning: Articles per page dropdown failed ({e}). Proceeding with default settings.")
    # ------------------------------------------------------------

    print(f"\nStarting targeted scrape for ALL EXTERNAL articles (Stopping after {MAX_EXTERNAL_TO_SCRAPE} collected)...")
    external_links_found_count = 0

    # --- 2. PAGINATION LOOP: Use "Show More" until end ---
    while True:
        
        # --- A. Scrape Articles Currently Visible ---
        article_containers = driver.find_elements(By.CSS_SELECTOR, working_article_selector)
        initial_container_count = len(article_containers)
        successful_external_scrapes_in_batch = 0
        
        # Iterate from the last processed index up to the current total
        for i in range(last_processed_index, initial_container_count):
                
            article = article_containers[i]
            
            # Skip the first article only on the initial load (i=0)
            if i == 0 and last_processed_index == 0:
                print("⏭️ Skipping the first article as requested.")
                last_processed_index = i + 1
                continue
                    
            # Extract basic metadata for filtering
            title = None
            link = None

            try:
                title, link = extract_title_and_link(article)
                if not title or not link:
                    last_processed_index = i + 1
                    continue
                
                # --- FILTERING STEP: ONLY PROCESS EXTERNAL LINKS ---
                if link.startswith(DOMAIN_ROOT):
                    last_processed_index = i + 1
                    continue
                
                # Increment external link counter regardless of successful scrape
                external_links_found_count += 1
                
                # --- METADATA EXTRACTION (from Brookings search page) ---
                authors = "Unknown"
                date = "Unknown"
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
                
                # --- CHECK IF MAX REACHED ---
                if len(data) >= MAX_EXTERNAL_TO_SCRAPE:
                    # We found an external link, but we're already full. Skip processing it.
                    last_processed_index = i + 1
                    continue
                
                # --- CONTENT EXTRACTION (SLOW STEP) ---
                print(f"\n[External {len(data) + 1}] Fetching content for: {title}")
                full_text = get_external_article_content(driver, link)
                
                if full_text:
                    data.append({
                        "Title": title,
                        "Authors": authors,
                        "Date": date,
                        "URL": link,
                        "External Full Text": full_text
                    })
                    successful_external_scrapes_in_batch += 1
                
            except Exception as e:
                print(f"Unhandled error processing article {i}: {e}. Skipping.")
                continue 
                
            # Update the last processed index for the next loop iteration
            last_processed_index = i + 1


        # --- C. Stop Condition Check ---
        if len(data) >= MAX_EXTERNAL_TO_SCRAPE:
            print(f"\n🛑 Reached maximum limit of {MAX_EXTERNAL_TO_SCRAPE} successfully scraped external articles. Stopping crawl.")
            break
        
        # --- D. Click "Show More" ---
        
        # If we processed all visible containers in this batch, attempt to click 'Show More'.
        if last_processed_index == initial_container_count and initial_container_count > 0:
            
            try:
                print(f"\n--- Processed {initial_container_count} articles. Attempting to click 'Show More'...")
                
                # Use a specific WebDriverWait to find the button
                wait = WebDriverWait(driver, 10) 
                show_more_button = wait.until(
                    EC.element_to_be_clickable((By.XPATH, SHOW_MORE_BUTTON_XPATH))
                )
                
                # Scroll and click
                driver.execute_script("arguments[0].scrollIntoView(true);", show_more_button)
                time.sleep(1)
                show_more_button.click()
                time.sleep(5) 
                
                # Check if the number of articles on the page increased after the click.
                new_container_count = len(driver.find_elements(By.CSS_SELECTOR, working_article_selector))
                
                if new_container_count > initial_container_count:
                    # New content loaded successfully, reset index to resume scraping from the old end point.
                    last_processed_index = initial_container_count
                    print(f"✅ Successfully clicked 'Show More'. Total containers now: {new_container_count}")
                    
                else:
                    print("🛑 'Show More' clicked, but no new articles loaded (End of results). Stopping crawl.")
                    break


            except TimeoutException:
                print("\n🛑 Timeout: 'Show More' button not found or clicked. Assuming end of results. Stopping crawl.")
                break
            except Exception as e:
                print(f"\n🛑 Error clicking 'Show More': {e}. Stopping crawl.")
                break
        
        elif initial_container_count == 0:
            print("\n🛑 No articles found on the first page. Stopping.")
            break
        
        else:
            print("\n🛑 Processed all available visible content without loading new pages. Stopping.")
            break


    driver.quit()
    df = pd.DataFrame(data)
    
    print("\n" + "="*50)
    print("EXTERNAL CONTENT SCRAPE COMPLETE")
    print("="*50)
    print(f"Total external links encountered: {external_links_found_count}")
    print(f"Collected and scraped: {len(df)} external records.")
    df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8-sig')
    print(f"Data saved to **{OUTPUT_FILENAME}**")

if __name__ == "__main__":
    scrape_external_content()