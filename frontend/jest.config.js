const nextJest = require("next/jest");

const createJestConfig = nextJest({ dir: "./" });

module.exports = async () => ({
  projects: [
    await createJestConfig({
      displayName: "node",
      testEnvironment: "node",
      testMatch: ["<rootDir>/__tests__/utils/**/*.test.ts"],
    })(),
    await createJestConfig({
      displayName: "jsdom",
      testEnvironment: "jsdom",
      testMatch: [
        "<rootDir>/__tests__/components/**/*.test.tsx",
        "<rootDir>/__tests__/pages/**/*.test.tsx",
        "<rootDir>/__tests__/lib/**/*.test.ts",
        "<rootDir>/__tests__/lib/**/*.test.tsx",
      ],
      setupFilesAfterEnv: ["<rootDir>/jest.setup.js"],
    })(),
  ],
});
