import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC 
from webdriver_manager.chrome import ChromeDriverManager 
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException

def scrape_hudson_articles_full_run_fixed(url):
    """
    Scrapes all article data from a Hudson Institute search results page, 
    navigating through all subsequent pages until the "Next" button is no longer found 
    OR the page fails to advance.
    """
    
    # --- 1. Setup WebDriver and Options ---
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("start-maximized")
    
    # Define XPaths
    BASE_CONTAINER_XPATH = "/html/body/div[1]/div/main/div/div/div[2]/article/div/div/div/div[4]/div/div/div/div"
    NEXT_BUTTON_XPATH = "/html/body/div[1]/div/main/div/div/div[2]/article/div/div/div/div[4]/div/div/div/nav/ul/li[12]/a"
    CONTENT_XPATH = "/html/body/div[1]/div/main/div/div/div[2]/article/div/div[2]/div/div[2]/div/div[1]"
    
    # XPath for the link of the first article (used for duplication check)
    FIRST_ARTICLE_LINK_XPATH = f"{BASE_CONTAINER_XPATH}/article/div/div/div[2]/a"


    driver = None
    all_scraped_data = []
    current_page = 1
    last_first_article_url = None # Variable to store the URL of the first article on the previous page
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(url)
        
        wait = WebDriverWait(driver, 20) 
        
        # Initial wait for the first article to load
        wait.until(EC.presence_of_element_located((By.XPATH, f"{BASE_CONTAINER_XPATH}/article")))
        print(f"Successfully loaded initial page: {url}")
        
        main_window = driver.current_window_handle
        
        # Main scraping loop
        while True:
            print(f"\n--- Scraping Page {current_page} (Total scraped: {len(all_scraped_data)}) ---")
            
            # Use a short delay for content stability
            time.sleep(2) 
            
            # --- 2. Locate All Articles and Check for Duplicates ---
            try:
                # Get the URL of the first article on the current page for duplicate detection
                first_article_element = wait.until(
                    EC.presence_of_element_located((By.XPATH, FIRST_ARTICLE_LINK_XPATH))
                )
                current_first_article_url = first_article_element.get_attribute('href')
            except (NoSuchElementException, TimeoutException):
                # If we can't even find the first article, something is broken or we hit a blank results page
                print("Could not find any articles on the current page. Breaking loop.")
                break
                
            # Check if we are stuck on the same page
            if last_first_article_url and current_first_article_url == last_first_article_url:
                print("🚨 Duplicate content detected. Page did not advance. Breaking loop to prevent infinite scrape.")
                break
                
            # Update the last recorded URL
            last_first_article_url = current_first_article_url
            
            article_elements = driver.find_elements(By.XPATH, f"{BASE_CONTAINER_XPATH}/article")
            print(f"Found {len(article_elements)} articles on this page.")
            
            # Iterate through all articles found on the current page
            for article_element in article_elements:
                
                # The elements are found relative to the article_element
                try:
                    # Scrape header info using relative XPaths
                    title_element = article_element.find_element(By.XPATH, ".//div/div/div[2]/a/span")
                    author_element = article_element.find_element(By.XPATH, ".//div/div/div[2]/div[2]/div/div[2]/div")
                    date_element = article_element.find_element(By.XPATH, ".//div/div/div[2]/div[3]/div[1]/div/time")
                    link_element = article_element.find_element(By.XPATH, ".//div/div/div[2]/a")

                    title = title_element.text
                    author = author_element.text
                    date = date_element.text
                    article_url = link_element.get_attribute('href')
                    
                    # --- 3. Scrape Full Content from Article Page (in new tab) ---
                    full_content = ""
                    
                    # Open the link in a new tab
                    driver.execute_script("window.open(arguments[0]);", article_url)
                    driver.switch_to.window(driver.window_handles[-1])
                    
                    try:
                        # Wait for the main content element to be present
                        content_wait = WebDriverWait(driver, 10)
                        content_element = content_wait.until(
                            EC.presence_of_element_located((By.XPATH, CONTENT_XPATH))
                        )
                        full_content = content_element.text
                    except (NoSuchElementException, TimeoutException):
                        # This is a common error on internal links; handle and continue
                        print(f"  > Could not find content for: '{title}'.")
                    
                    # Close the article tab and switch back to the search page tab
                    driver.close()
                    driver.switch_to.window(main_window)
                    
                    all_scraped_data.append({
                        'Title': title,
                        'Author': author,
                        'Date': date,
                        'URL': article_url,
                        'Content': full_content.strip() 
                    })
                    print(f"  > Scraped article #{len(all_scraped_data)}: {title}")

                except (NoSuchElementException, TimeoutException, StaleElementReferenceException) as e:
                    print(f"  > Skipping an article due to element error: {e.__class__.__name__}")
                    continue 

            # --- 4. Pagination Logic: Find and Click Next Button ---
            try:
                # Wait for the next button to be visible and clickable
                next_button = wait.until(
                    EC.element_to_be_clickable((By.XPATH, NEXT_BUTTON_XPATH))
                )
                
                # Execute click via JavaScript
                driver.execute_script("arguments[0].click();", next_button)
                current_page += 1
                
                # Wait for the first article on the *new* page to load before continuing
                wait.until(
                    EC.presence_of_element_located((By.XPATH, FIRST_ARTICLE_LINK_XPATH))
                )

            except (NoSuchElementException, TimeoutException):
                # This is the final page condition: Next button is no longer present/clickable
                print("\nNext button not found or not clickable via XPath. Assumed end of pagination.")
                break # Exit the while loop
                
        # --- 5. Create and Return DataFrame ---
        df = pd.DataFrame(all_scraped_data)
        return df

    except Exception as e:
        print(f"\nAn unexpected error occurred during WebDriver operation: {e}")
        return pd.DataFrame()
    finally:
        if driver:
            print("\nClosing the browser...")
            driver.quit()

# --- Execution ---
if __name__ == '__main__':
    TARGET_URL = "https://www.hudson.org/search?keywords=taiwan"
    
    print("🚀 Starting the Hudson Institute scraper with duplicate content detection...")
    data_frame = scrape_hudson_articles_full_run_fixed(TARGET_URL)
    
    if not data_frame.empty:
        print(f"\n✅ Scraping complete. Total articles scraped: {len(data_frame)}")
        print("\n--- First 5 Scraped Records (Title, Author, Date, URL) ---")
        print(data_frame[['Title', 'Author', 'Date', 'URL']].head())
        
        # Save to CSV
        file_name = 'hudson_taiwan_articles_all_fixed.csv'
        data_frame.to_csv(file_name, index=False)
        print(f"\nData saved to '{file_name}'")
    else:
        print("\n❌ Scraping failed or no data was collected.")
