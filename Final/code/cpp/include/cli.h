#ifndef CLI_H
#define CLI_H

class Config {
public:
    Config(int argc, char **argv);
    void apply() const;

    enum class Method {
        UzawaLDL,
        UzawaCG,
        UzawaPCG,
        MultiGrid,
    };

private:
    int maxThreads;
    Method method;
};

#endif // CLI_H
