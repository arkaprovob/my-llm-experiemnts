import requests
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# The URL you want to check
url = ('https://route-spaship-manager-dev-preprod.apps.ext-waf.mpp.preprod.iad2.dc.paas.redhat.com'
       '/lctx/chunks/long-context/10mb.txt')
check_content_starts_with = 'Hello World'
break_on_different_content = True


def is_a_different_content(response_text, test_with) -> bool:
    if not response_text.startswith(test_with):
        return True
    return False


# Counter to keep track of iterations
counter = 0

while True:
    # Increment the counter
    counter += 1

    try:
        # Sending a GET request with 'Cache-Control' header set to 'no-cache'
        response = requests.get(url, headers={'Cache-Control': 'no-cache'})
    except requests.exceptions.RequestException as e:
        logging.error(f'{datetime.now()}: Error: {e}')
        break

    # Checking if the content is blank
    if response.text.strip() == '':
        logging.warning(f'{datetime.now()}: Warning: The content is blank! (found at iteration '
                        f'#{counter})')
        break  # Exit the loop if the content is blank
    else:
        if is_a_different_content(response.text, check_content_starts_with):
            logging.info(f'{datetime.now()}: a different content found.')
            if break_on_different_content:
                logging.info(f'{datetime.now()}: Breaking on different content. at iteration #'
                             f'{counter}')
                break
        logging.info(f'{datetime.now()}: Successful: The content is not blank.')
