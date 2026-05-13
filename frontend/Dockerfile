FROM node:20-alpine

WORKDIR /app

# package.json появится после ручного `npm create vite@latest`
# (см. frontend/SETUP.md). При первой dev-сборке скрипт init в compose
# выполняет npm install — здесь это закомментировано, потому что на момент
# build файла может не быть.
COPY package*.json ./
RUN if [ -f package.json ]; then npm install; fi

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
